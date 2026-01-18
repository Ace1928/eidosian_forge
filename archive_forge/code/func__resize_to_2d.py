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
def _resize_to_2d(self, x: torch.Tensor):
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    return x