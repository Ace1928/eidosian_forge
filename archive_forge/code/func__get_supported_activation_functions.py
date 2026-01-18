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
def _get_supported_activation_functions():
    SUPPORTED_ACTIVATION_FUNCTIONS = {F.relu, F.rrelu, F.hardtanh, F.relu6, F.sigmoid, F.hardsigmoid, F.tanh, F.silu, F.mish, F.hardswish, F.elu, F.celu, F.selu, F.hardshrink, F.leaky_relu, F.logsigmoid, F.softplus, F.prelu, F.softsign, F.tanhshrink, F.gelu}
    return SUPPORTED_ACTIVATION_FUNCTIONS