import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldStructureModuleTransition(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for _ in range(config.num_transition_layers):
            l = EsmFoldStructureModuleTransitionLayer(config)
            self.layers.append(l)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = LayerNorm(config.sequence_dim)

    def forward(self, s):
        for l in self.layers:
            s = l(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s