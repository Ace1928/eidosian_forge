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
class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = PatchTSMixerMLP(in_features=config.num_input_channels, out_features=config.num_input_channels, config=config)
        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.num_input_channels, out_size=config.num_input_channels)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)
        inputs = inputs.permute(0, 3, 2, 1)
        if self.gated_attn:
            inputs = self.gating_block(inputs)
        inputs = self.mlp(inputs)
        inputs = inputs.permute(0, 3, 2, 1)
        out = inputs + residual
        return out