import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        if config.feat_extract_norm == 'group':
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)]
        elif config.feat_extract_norm == 'layer':
            conv_layers = [SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)
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