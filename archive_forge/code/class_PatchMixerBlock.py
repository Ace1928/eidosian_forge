import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(config)
        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn
        self.mlp = PatchTSMixerMLP(in_features=config.num_patches, out_features=config.num_patches, config=config)
        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.num_patches, out_size=config.num_patches)
        if config.self_attn:
            self.self_attn_layer = PatchTSMixerAttention(embed_dim=config.d_model, num_heads=config.self_attn_heads, dropout=config.dropout)
            self.norm_attn = PatchTSMixerNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)
            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)
        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)
        hidden_state = hidden_state.transpose(2, 3)
        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)
        out = hidden_state + residual
        return out