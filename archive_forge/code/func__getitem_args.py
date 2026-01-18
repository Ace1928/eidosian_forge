import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def _getitem_args(self, getitem_node: torch.fx.Node):
    if getitem_node.target != operator.__getitem__ or getitem_node.op != 'call_function':
        return None
    return getitem_node.args[0]