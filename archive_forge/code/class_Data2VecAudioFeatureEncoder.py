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
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_data2vec_audio import Data2VecAudioConfig
class Data2VecAudioFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.ModuleList([Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)])
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(conv_layer.__call__, hidden_states)
            else:
                hidden_states = conv_layer(hidden_states)
        return hidden_states