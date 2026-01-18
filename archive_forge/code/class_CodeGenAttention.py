from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_codegen import CodeGenConfig
class CodeGenAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        max_positions = config.max_position_embeddings
        self.register_buffer('causal_mask', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions), persistent=False)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads}).')
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f'Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}')
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query_length, key_length = (query.size(-2), key.size(-2))
        causal_mask = self.causal_mask[:, :, key_length - query_length:key_length, :key_length]
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def forward(self, hidden_states: Optional[torch.FloatTensor], layer_past: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]]]:
        qkv = self.qkv_proj(hidden_states)
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        sincos = embed_positions[position_ids]
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, :self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]
            q_rot = query[:, :, :, :self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]
            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs