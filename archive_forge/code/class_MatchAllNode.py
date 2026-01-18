import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
class MatchAllNode:
    """ A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """
    pass