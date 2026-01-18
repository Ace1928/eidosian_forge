import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
class FlavaImageCodebookBlock(nn.Module):

    def __init__(self, in_size: int, out_size: int, num_layers: int, **kwargs):
        super().__init__()
        self.post_gain = 1 / num_layers ** 2
        if in_size != out_size:
            self.id_path = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        else:
            self.id_path = nn.Identity()
        self.res_path = FlavaImageCodebookResPath(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)