import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
class Conv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, config):
        super(Conv1dSubsampler, self).__init__()
        self.config = config
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
        self.mid_channels = config.conv_channels
        self.out_channels = config.d_model
        self.kernel_sizes = config.conv_kernel_sizes
        self.conv_layers = nn.ModuleList((nn.Conv1d(self.in_channels if i == 0 else self.mid_channels // 2, self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2, kernel_size=k, stride=2, padding=k // 2) for i, k in enumerate(self.kernel_sizes)))

    def forward(self, input_features):
        hidden_states = input_features.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.glu(hidden_states, dim=1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        return hidden_states