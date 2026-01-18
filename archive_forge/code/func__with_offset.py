from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def _with_offset(func: Callable[..., None]) -> Callable[..., None]:
    nonlocal _crit_path_counter_offset

    def wrapper(*args: Any, **kwargs: Any) -> None:
        nonlocal _crit_path_counter_offset
        _crit_path_counter_offset = 0.5
        try:
            func(*args, **kwargs)
        finally:
            _crit_path_counter_offset = 0
    return wrapper