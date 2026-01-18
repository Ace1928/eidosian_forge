import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def _is_input_2d(self, input: torch.fx.Node) -> bool:
    return len(input.meta['tensor_meta'].shape) == 2