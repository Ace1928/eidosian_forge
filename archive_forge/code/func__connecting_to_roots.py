from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def _connecting_to_roots(dependencies: Mapping[Key, set[Key]], dependents: Mapping[Key, set[Key]]) -> tuple[dict[Key, set[Key]], dict[Key, int]]:
    """Determine for every node which root nodes are connected to it (i.e.
    ancestors). If arguments of dependencies and dependents are switched, this
    can also be used to determine which leaf nodes are connected to which node
    (i.e. descendants).

    Also computes a weight that is defined as (cheaper to compute here)

            `max(len(dependents[k]) for k in connected_roots[key])`

    """
    result = {}
    current = []
    num_needed = {k: len(v) for k, v in dependencies.items() if v}
    max_dependents = {}
    roots = set()
    for k, v in dependencies.items():
        if not v:
            roots.add(k)
            result[k] = {k}
            deps = dependents[k]
            max_dependents[k] = len(deps)
            for child in deps:
                num_needed[child] -= 1
                if not num_needed[child]:
                    current.append(child)
    while current:
        key = current.pop()
        for parent in dependents[key]:
            num_needed[parent] -= 1
            if not num_needed[parent]:
                current.append(parent)
        new_set = None
        identical_sets = True
        result_first = None
        for child in dependencies[key]:
            r_child = result[child]
            if not result_first:
                result_first = r_child
                max_dependents[key] = max_dependents[child]
            elif not (identical_sets and (result_first is r_child or r_child.issubset(result_first))):
                identical_sets = False
                if not new_set:
                    new_set = result_first.copy()
                max_dependents[key] = max(max_dependents[child], max_dependents[key])
                new_set.update(r_child)
        assert new_set is not None or result_first is not None
        result[key] = new_set or result_first
    empty_set: set[Key] = set()
    for r in roots:
        result[r] = empty_set
    return (result, max_dependents)