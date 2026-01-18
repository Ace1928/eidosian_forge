from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def _convert_task(task: Any) -> Any:
    if istask(task):
        assert callable(task[0])
        new_spec: list[Any] = []
        for el in task[1:]:
            if isinstance(el, (str, int)):
                new_spec.append(el)
            elif isinstance(el, tuple):
                if istask(el):
                    new_spec.append(_convert_task(el))
                else:
                    new_spec.append(el)
            elif isinstance(el, list):
                new_spec.append([_convert_task(e) for e in el])
        return (_f, *new_spec)
    elif isinstance(task, tuple):
        return (_f, task)
    else:
        return (_f, *task)