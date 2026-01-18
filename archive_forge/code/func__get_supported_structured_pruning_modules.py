from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import (
def _get_supported_structured_pruning_modules():
    SUPPORTED_STRUCTURED_PRUNING_MODULES = {nn.Linear, nn.Conv2d, nn.LSTM}
    return SUPPORTED_STRUCTURED_PRUNING_MODULES