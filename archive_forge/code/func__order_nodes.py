import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def _order_nodes(node_a, node_b, node_c) -> List[Node]:
    nodes = [node_a, node_b, node_c]
    first_node = None
    mid_node = None
    last_node = None
    for n in nodes:
        prev_n = n.args[0]
        next_n = next(iter(n.users))
        if prev_n not in nodes:
            first_node = n
        elif next_n not in nodes:
            last_node = n
        else:
            mid_node = n
    assert first_node is not None and mid_node is not None and (last_node is not None)
    assert mid_node.args[0] is first_node
    assert last_node.args[0] is mid_node
    return [last_node, mid_node, first_node]