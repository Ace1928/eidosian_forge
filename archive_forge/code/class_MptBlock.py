import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mpt import MptConfig
class MptBlock(nn.Module):

    def __init__(self, config: MptConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.norm_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.norm_1.bias = None
        self.num_heads = config.n_heads
        self.attn = MptAttention(config)
        self.norm_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.norm_2.bias = None
        self.ffn = MptMLP(config)
        self.dropout_rate = config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: bool=False, output_attentions: bool=False):
        layernorm_output = self.norm_1(hidden_states)
        residual = hidden_states
        attn_outputs, attn_weights, past_key_value = self.attn(layernorm_output, position_bias=position_bias, attention_mask=attention_mask, past_key_value=layer_past)
        hidden_states = self.resid_attn_dropout(attn_outputs) + residual
        layernorm_output = self.norm_2(hidden_states)
        residual = hidden_states
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)
        if use_cache:
            outputs += (past_key_value,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs