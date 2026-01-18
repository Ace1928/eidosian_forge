import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _insert_context(self, func: str, node: fx.Node):

    class _InsertPoint:

        def __init__(self, insert_points: List[Any]):
            self.insert_points = insert_points

        def __enter__(self):
            pass

        def __exit__(self, type, value, tb):
            for insert_point in self.insert_points:
                insert_point.__exit__(type, value, tb)
    insert_points = []
    for graph in self._all_graphs:
        if node:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, 'Cannot handle None case now.'
        else:
            actual_node = node
        insert_points.append(getattr(graph, func)(actual_node))
    return _InsertPoint(insert_points)