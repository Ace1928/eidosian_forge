import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def _addmm_node_can_be_fused(self, node: torch.fx.Node):
    input_shape = node.args[1].meta['tensor_meta'].shape
    weight_shape = node.args[2].meta['tensor_meta'].shape
    return node.kwargs.get('beta', 1.0) == 1.0 and node.kwargs.get('alpha', 1.0) == 1.0 and (len(input_shape) == 2) and (len(weight_shape) == 2) and all((x % 2 == 0 for x in input_shape + weight_shape)) and all((shape <= self.graph_search_options['max_fuse_tensor_size_group_linear'] for shape in input_shape + weight_shape))