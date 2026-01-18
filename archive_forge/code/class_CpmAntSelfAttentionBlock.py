import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
class CpmAntSelfAttentionBlock(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.layernorm_before_attention = CpmAntLayerNorm(config)
        self.self_attention = CpmAntAttention(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_bias: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: Optional[bool]=None):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        outputs = self.layernorm_before_attention(hidden_states)
        outputs = self.self_attention(outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache)
        outputs, attn_weights, current_key_value = outputs
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs
        return (hidden_states, attn_weights, current_key_value)