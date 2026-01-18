import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_nat import NatConfig
class NatStage(nn.Module):

    def __init__(self, config, dim, depth, num_heads, drop_path_rate, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layers = nn.ModuleList([NatLayer(config=config, dim=dim, num_heads=num_heads, drop_path_rate=drop_path_rate[i]) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
        self.pointing = False

    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        _, height, width, _ = hidden_states.size()
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling)
        stage_outputs = (hidden_states, hidden_states_before_downsampling)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs