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
class Data2VecAudioAdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.output_hidden_size, 2 * config.output_hidden_size, config.adapter_kernel_size, stride=config.adapter_stride, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states