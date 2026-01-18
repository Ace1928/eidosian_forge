from __future__ import annotations
from collections.abc import Callable
from itertools import zip_longest
from numbers import Integral
from typing import Any
import numpy as np
from dask import config
from dask.array.chunk import getitem
from dask.array.core import getter, getter_inline, getter_nofancy
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten, reverse_dict
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse, inline_functions
from dask.utils import ensure_dict
def _is_getter_task(value) -> tuple[Callable, Any, Any, bool, bool | None] | None:
    """Check if a value in a Dask graph looks like a getter.

    1. Is it a tuple with the first element a known getter.
    2. Is it a SubgraphCallable with a single element in its
       dsk which is a known getter.

    If a getter is found, it returns a tuple with (getter, array, index, asarray, lock).
    Otherwise it returns ``None``.

    TODO: the second check is a hack to allow for slice fusion between tasks produced
    from blockwise layers and slicing operations. Once slicing operations have
    HighLevelGraph layers which can talk to Blockwise layers this check *should* be
    removed, and we should not have to introspect SubgraphCallables.
    """
    if type(value) is not tuple:
        return None
    first = value[0]
    get: Callable | None = None
    if first in GETTERS:
        get = first
    elif isinstance(first, SubgraphCallable) and len(first.dsk) == 1:
        v = next(iter(first.dsk.values()))
        if type(v) is tuple and len(v) > 1 and (v[0] in GETTERS):
            get = v[0]
    if get is None:
        return None
    length = len(value)
    if length == 3:
        return (get, value[1], value[2], get is not getitem, None)
    elif length == 5:
        return (get, *value[1:])
    return None