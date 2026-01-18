import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
class Wav2Vec2BertConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(config.hidden_size, 2 * config.hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(config.hidden_size, config.hidden_size, config.conv_depthwise_kernel_size, stride=1, padding=0, groups=config.hidden_size, bias=False)
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states