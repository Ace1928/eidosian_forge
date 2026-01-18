import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def _get_all_child_nodes(self) -> List['DAGNode']:
    """Return the list of nodes referenced by the args, kwargs, and
        args_to_resolve in current node, even they're deeply nested.

        Examples:
            f.remote(a, [b]) -> [a, b]
            f.remote(a, [b], key={"nested": [c]}) -> [a, b, c]
        """
    scanner = _PyObjScanner()
    children = []
    for n in scanner.find_nodes([self._bound_args, self._bound_kwargs, self._bound_other_args_to_resolve]):
        if n not in children:
            children.append(n)
    scanner.clear()
    return children