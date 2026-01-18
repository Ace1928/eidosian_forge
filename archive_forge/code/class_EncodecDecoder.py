import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [EncodecConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]
        model += [EncodecLSTM(config, scaling * config.num_filters)]
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            model += [EncodecConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)]
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale // 2, (config.dilation_growth_rate ** j, 1))]
            scaling //= 2
        model += [nn.ELU()]
        model += [EncodecConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states