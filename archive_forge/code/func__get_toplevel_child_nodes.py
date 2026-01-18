import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def _get_toplevel_child_nodes(self) -> List['DAGNode']:
    """Return the list of nodes specified as top-level args.

        For example, in `f.remote(a, [b])`, only `a` is a top-level arg.

        This list of nodes are those that are typically resolved prior to
        task execution in Ray. This does not include nodes nested within args.
        For that, use ``_get_all_child_nodes()``.
        """
    children = []
    for a in self.get_args():
        if isinstance(a, DAGNode):
            if a not in children:
                children.append(a)
    for a in self.get_kwargs().values():
        if isinstance(a, DAGNode):
            if a not in children:
                children.append(a)
    for a in self.get_other_args_to_resolve().values():
        if isinstance(a, DAGNode):
            if a not in children:
                children.append(a)
    return children