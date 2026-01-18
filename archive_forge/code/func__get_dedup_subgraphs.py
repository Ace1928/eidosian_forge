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
def _get_dedup_subgraphs(matches: Dict[str, _MatchResult]) -> Dict[str, List[Node]]:
    seen_nodes = set()
    subgraphs_dedup = {}
    matches_items_reversed: List[Tuple[str, _MatchResult]] = []
    for name, cur_match in matches.items():
        matches_items_reversed.insert(0, (name, cur_match))
    for name, cur_match in matches_items_reversed:
        was_seen = False
        for node_or_tuple in cur_match[1]:
            if isinstance(node_or_tuple, Node):
                if node_or_tuple in seen_nodes:
                    was_seen = True
                seen_nodes.add(node_or_tuple)
            else:
                assert isinstance(node_or_tuple, tuple)
                for node in node_or_tuple:
                    assert isinstance(node, Node)
                    if node in seen_nodes:
                        was_seen = True
                    seen_nodes.add(node)
        if was_seen:
            continue
        list_of_nodes = []
        if len(cur_match[1]) == 1:
            list_of_nodes = cur_match[1]
        else:
            assert len(cur_match[1]) == 2

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
            if isinstance(cur_match[1][0], Node) and isinstance(cur_match[1][1], Node):
                list_of_nodes = cur_match[1]
            elif isinstance(cur_match[1][0], tuple):
                node_a, node_b = cur_match[1][0]
                node_c = cur_match[1][1]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
            elif isinstance(cur_match[1][1], tuple):
                node_a, node_b = cur_match[1][1]
                node_c = cur_match[1][0]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
        list_of_nodes.reverse()
        subgraphs_dedup[name] = list_of_nodes
    return subgraphs_dedup