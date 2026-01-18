import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
def _topological_sort_passes(passes: List[Callable], constraints: List[Callable]) -> List[Callable]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    if len(constraints) == 0:
        return passes
    graph: Dict[Callable, List[Callable]] = {p: [] for p in passes}
    indegree_map: Dict[Callable, int] = {p: 0 for p in passes}
    candidates: Queue = Queue()
    for a in passes:
        for b in passes:
            if a == b:
                continue
            for constraint in constraints:
                if not constraint(a, b):
                    graph[b].append(a)
                    indegree_map[a] += 1
        if indegree_map[a] == 0:
            candidates.put(a)
    visited: Dict[Callable, bool] = {p: False for p in passes}
    sorted_passes: List[Callable] = []
    while not candidates.empty():
        p = candidates.get()
        sorted_passes.append(p)
        visited[p] = True
        for n in graph[p]:
            if not visited[n]:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)
    cycle_passes = list(filter(lambda p: indegree_map[p] != 0, indegree_map.keys()))
    if len(cycle_passes) != 0:
        error = f'Circular dependency detected within the following passes: {cycle_passes}'
        raise RuntimeError(error)
    return sorted_passes