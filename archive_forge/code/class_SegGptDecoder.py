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
class SegGptDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size * len(config.intermediate_hidden_state_indices), config.patch_size ** 2 * config.decoder_hidden_size, bias=True)
        self.decoder_pred = SegGptDecoderHead(config)
        self.patch_size = config.patch_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.config = config

    def _reshape_hidden_states(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, patch_height, patch_width, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, patch_height, patch_width, self.patch_size, self.patch_size, self.decoder_hidden_size)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(shape=(batch_size, -1, patch_height * self.patch_size, patch_width * self.patch_size))
        return hidden_states

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.decoder_embed(hidden_states)
        hidden_states = self._reshape_hidden_states(hidden_states)
        hidden_states = self.decoder_pred(hidden_states)
        return hidden_states