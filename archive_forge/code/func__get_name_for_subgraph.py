import collections
import enum
import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import (
from .pattern_utils import (
from torch.ao.quantization import (
from typing import Dict, Tuple, List, Optional, Set, Any
def _get_name_for_subgraph(subgraph_a: NSSubgraph, gm_a: GraphModule, base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]], existing_names: Set[str]) -> str:
    """
    Returns a unique name for a subgraph. This name is based on two things:
    1. the name of the set containing the underlying type of the base op in the
       subgraph (i.e. 'torch.nn.functional.linear' if this is related to a linear op)
    2. the number of previous subgraphs with related underlying type of the base op

    For example, in the graph

    linear0 -> relu0 -> linear1 -> relu1

    The subgraphs are (linear0, relu0) and (linear1, relu1).  If we iterate
    from the output node backwards, the name given to (linear1, relu1) will be
    `base_op_torch.nn.functional.linear_0`, and the name given to (linear0, relu0)
    will be `base_op_torch.nn.functional.linear_1`.

    Why are we not just using the node name? Answer: because of two requirements:
    A. fusions must be supported
    B. some Numeric Suite APIs can be called without having all of the models in memory

    For example, let's say we need to match nodes of

    (1) ... -> linear0 -> relu0 -> ...

    And

    (2) ... -> linear_relu0 -> ...

    Without being able to inspect them together. With the current naming scheme, if
    we iterate through both of these graphs in the same order, and assuming the rest
    of the graphs match, both of these subgraphs will get the same name without
    (1) and (2) knowing anything about each other.
    """
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    for base_name, sets_of_related_ops in base_name_to_sets_of_related_ops.items():
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    target_base_name = 'base_op_' + str(target_base_type)
    counter = 0
    proposed_name = target_base_name + '_' + str(counter)
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + '_' + str(counter)
    existing_names.add(proposed_name)
    return proposed_name