import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def _get_split_args_default(split_node):
    input_kwarg = 'tensor'
    split_size_kwarg = 'split_size_or_sections'
    dim_kwarg = 'dim'
    default_dim_value = 0
    if split_node.op == 'call_method':
        split_size_kwarg = 'split_size'
    return (get_arg_value(split_node, 0, input_kwarg), get_arg_value(split_node, 1, split_size_kwarg), get_arg_value(split_node, 2, dim_kwarg) or default_dim_value)