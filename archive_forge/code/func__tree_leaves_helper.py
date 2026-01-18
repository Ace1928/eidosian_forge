import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _tree_leaves_helper(tree: PyTree, leaves: List[Any]) -> None:
    if _is_leaf(tree):
        leaves.append(tree)
        return
    node_type = _get_node_type(tree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, _ = flatten_fn(tree)
    for child in child_pytrees:
        _tree_leaves_helper(child, leaves)