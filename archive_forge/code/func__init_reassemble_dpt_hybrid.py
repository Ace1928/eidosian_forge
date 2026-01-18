import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
def _init_reassemble_dpt_hybrid(self, config):
    """ "
        For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
    for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
        if i <= 1:
            self.layers.append(nn.Identity())
        elif i > 1:
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))
    if config.readout_type != 'project':
        raise ValueError(f'Readout type {config.readout_type} is not supported for DPT-Hybrid.')
    self.readout_projects = nn.ModuleList()
    hidden_size = _get_backbone_hidden_size(config)
    for i in range(len(config.neck_hidden_sizes)):
        if i <= 1:
            self.readout_projects.append(nn.Sequential(nn.Identity()))
        elif i > 1:
            self.readout_projects.append(nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act]))