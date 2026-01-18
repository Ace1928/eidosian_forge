import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2ConvFeedForwardNetwork(nn.Module):

    def __init__(self, config: PvtV2Config, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = PvtV2DepthWiseConv(config, hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU() if config.linear_attention else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, height, width) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states