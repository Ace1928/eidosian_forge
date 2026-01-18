import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gpt_bigcode import GPTBigCodeConfig
class GPTBigCodeSdpaAttention(GPTBigCodeAttention):

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if head_mask is not None:
            raise ValueError('PyTorch SDPA does not support head_mask. Please open an issue in Transformers repository.')
        scale = None
        if not self.scale_attn_weights:
            scale = 1
        query_shape = query.shape
        batch_size = query_shape[0]
        key.shape[-2]
        if self.multi_query:
            query_length = query_shape[1]
            query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        else:
            query_length = query_shape[-1]
            if query.device.type == 'cuda' and attention_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=self.attn_pdrop if self.training else 0.0, is_causal=self.is_causal and attention_mask is None and (query_length > 1), scale=scale)
        if self.multi_query:
            sdpa_result = sdpa_result.transpose(1, 2)
            sdpa_result = sdpa_result.reshape(query_shape)
        return (sdpa_result, None)

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]]:
        if encoder_hidden_states is not None:
            if not hasattr(self, 'q_attn') or not self.is_cross_attention:
                raise ValueError('If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`.')
            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            query, key_value = self.c_attn(hidden_states).view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim).transpose(1, 2).split((self.head_dim, 2 * self.head_dim), dim=3)
        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None
        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)
        if not output_attentions and head_mask is None:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        else:
            logger.warning_once('GPTBigCodeModel is using GPTBigCodeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` and `head_mask` not None. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.')
            attn_output, attn_weights = super()._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)
        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)
        return outputs