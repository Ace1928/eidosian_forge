import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig
class PoolFormerOutput(nn.Module):

    def __init__(self, config, dropout_prob, hidden_size, intermediate_size):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, intermediate_size, 1)
        self.conv2 = nn.Conv2d(intermediate_size, hidden_size, 1)
        self.drop = PoolFormerDropPath(dropout_prob)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states