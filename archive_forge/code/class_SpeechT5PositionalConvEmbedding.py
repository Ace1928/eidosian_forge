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
class SpeechT5PositionalConvEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings, padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups)
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, 'weight_norm'):
            weight_norm = nn.utils.parametrizations.weight_norm
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name='weight', dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name='weight', dim=2)
        self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states