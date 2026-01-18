import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class SegGptDecoderHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(config.decoder_hidden_size, config.decoder_hidden_size, kernel_size=3, padding=1)
        self.layernorm = SegGptLayerNorm(normalized_shape=config.decoder_hidden_size, eps=config.layer_norm_eps, data_format='channels_first')
        self.act_fct = ACT2FN[config.hidden_act]
        self.head = nn.Conv2d(config.decoder_hidden_size, 3, kernel_size=1, bias=True)

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.act_fct(hidden_states)
        hidden_states = self.head(hidden_states)
        return hidden_states