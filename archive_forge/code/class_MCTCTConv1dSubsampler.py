import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTConv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.glu_dim = config.conv_glu_dim
        self.dropout = nn.Dropout(config.conv_dropout)
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
        if self.num_layers > 1:
            if config.conv_channels is None:
                raise ValueError('Need to specify `conv_channels` configuration in `MCTCTConfig` to use multiple convolution layers.')
            self.mid_channels = config.conv_channels
        else:
            self.mid_channels = None
        self.out_channels = config.hidden_size * 2
        self.kernel_size = config.conv_kernel
        self.stride = config.conv_stride
        self.conv_layers = nn.ModuleList((nn.Conv1d(self.in_channels if i == 0 else self.mid_channels[i], self.mid_channels[i] if i < self.num_layers - 1 else self.out_channels, kernel_size=k, stride=self.stride[i], padding='valid') for i, k in enumerate(self.kernel_size)))

    def forward(self, input_features):
        padding = sum([size // 2 for size in self.kernel_size])
        input_features = torch.nn.functional.pad(input_features, (0, 0, padding, padding), 'constant', 0)
        hidden_states = input_features.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.glu(hidden_states, dim=self.glu_dim)
            hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        return hidden_states