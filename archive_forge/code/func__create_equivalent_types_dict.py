import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node
import torch
from torch.fx.passes.utils.source_matcher_utils import (
def _create_equivalent_types_dict():
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT