import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
class RecurrentGemmaDecoderLayer(nn.Module):
    """Griffin and Hawk's residual block."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.temporal_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.temporal_block = TEMPORAL_BLOCK_CLASSES[config.layers_block_type[layer_idx]](config)
        self.channel_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = RecurrentGemmaMlp(config)

    def forward(self, activations: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor, cache_position: torch.Tensor=None, use_cache: bool=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_activations = activations
        inputs_normalized = self.temporal_pre_norm(raw_activations)
        hidden_states = self.temporal_block(inputs_normalized, position_ids, attention_mask, cache_position=cache_position, use_cache=use_cache)
        residual = hidden_states + raw_activations
        hidden_states = self.channel_pre_norm(residual)
        hidden_states = self.mlp_block(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states