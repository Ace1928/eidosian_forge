import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _tree_flatten_helper(tree: PyTree, leaves: List[Any]) -> TreeSpec:
    if _is_leaf(tree):
        leaves.append(tree)
        return _LEAF_SPEC
    node_type = _get_node_type(tree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(tree)
    children_specs = [_tree_flatten_helper(child, leaves) for child in child_pytrees]
    return TreeSpec(node_type, context, children_specs)