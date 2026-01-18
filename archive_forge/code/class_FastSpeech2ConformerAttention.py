import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerAttention(nn.Module):
    """
    Multi-Head attention layer with relative position encoding. Details can be found in
    https://github.com/espnet/espnet/pull/2816. Paper: https://arxiv.org/abs/1901.02860.
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """Construct an FastSpeech2ConformerAttention object."""
        super().__init__()
        self.num_heads = module_config['num_attention_heads']
        self.hidden_size = config.hidden_size
        self.dim_key = self.hidden_size // self.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=module_config['attention_dropout_rate'])
        self.linear_pos = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

    def shift_relative_position_tensor(self, pos_tensor):
        """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)
        pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
        pos_tensor = pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, :pos_tensor.size(-1) // 2 + 1]
        return pos_tensor

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pos_emb: Optional[torch.Tensor]=None, output_attentions: Optional[torch.Tensor]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time2, size)`): Values of the hidden states
            attention_mask (`torch.Tensor` of shape `(batch, time1, time2)`): Mask tensor.
            pos_emb (`torch.Tensor` of shape `(batch, 2*time1-1, size)`): Positional embedding tensor.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time1, d_model)`.
        """
        bsz, q_len, _ = hidden_states.size()
        query_states = self.linear_q(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        key_states = self.linear_k(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        value_states = self.linear_v(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        bsz_pos = pos_emb.size(0)
        pos_encoding = self.linear_pos(pos_emb).view(bsz_pos, -1, self.num_heads, self.head_dim)
        query_with_bias_u = (query_states + self.pos_bias_u).transpose(1, 2)
        query_with_bias_v = (query_states + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(query_with_bias_u, key_states.permute(0, 2, 3, 1))
        matrix_bd = torch.matmul(query_with_bias_v, pos_encoding.permute(0, 2, 3, 1))
        matrix_bd = self.shift_relative_position_tensor(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.dim_key)
        if attention_mask is not None:
            expected_size = (bsz, 1, q_len)
            if attention_mask.size() != expected_size:
                raise ValueError(f'Attention mask should be of size {expected_size}, but is {attention_mask.size()}')
            attention_mask = attention_mask.unsqueeze(1).eq(0)
            min_value = float(torch.finfo(scores.dtype).min)
            scores = scores.masked_fill(attention_mask, min_value)
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(attention_mask, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.linear_out(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights)