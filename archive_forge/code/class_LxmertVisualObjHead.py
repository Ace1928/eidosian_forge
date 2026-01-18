import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
class LxmertVisualObjHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses['obj'] = {'shape': (-1,), 'num': config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses['attr'] = {'shape': (-1,), 'num': config.num_attr_labels}
        if config.visual_feat_loss:
            visual_losses['feat'] = {'shape': (-1, config.visual_feat_dim), 'num': config.visual_feat_dim}
        self.visual_losses = visual_losses
        self.decoder_dict = nn.ModuleDict({key: nn.Linear(config.hidden_size, self.visual_losses[key]['num']) for key in self.visual_losses})

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output