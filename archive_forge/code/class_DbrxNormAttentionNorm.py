import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
class DbrxNormAttentionNorm(nn.Module):

    def __init__(self, config: DbrxConfig, block_idx: Optional[int]=None):
        super().__init__()
        self.block_idx = block_idx
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DBRX_ATTENTION_CLASSES[config._attn_implementation](config=config, block_idx=block_idx)
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states).to(hidden_states.dtype)
        hidden_states, attn_weights, past_key_value = self.attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, **kwargs)
        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = hidden_states + residual_states
        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states).to(hidden_states.dtype)
        return (residual_states, hidden_states, attn_weights, past_key_value)