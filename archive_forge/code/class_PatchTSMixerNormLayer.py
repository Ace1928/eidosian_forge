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
class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.norm_mlp = config.norm_mlp
        if 'batch' in config.norm_mlp.lower():
            self.norm = PatchTSMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if 'batch' in self.norm_mlp.lower():
            inputs_reshaped = torch.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            inputs_reshaped = self.norm(inputs_reshaped)
            inputs = torch.reshape(inputs_reshaped, inputs.shape)
        else:
            inputs = self.norm(inputs)
        return inputs