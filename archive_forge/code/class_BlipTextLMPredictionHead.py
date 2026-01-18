import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_blip import BlipTextConfig
class BlipTextLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BlipTextPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states