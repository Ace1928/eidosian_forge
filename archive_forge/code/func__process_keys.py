from __future__ import annotations
from collections.abc import Hashable, Iterable, Generator
import copy
from functools import reduce
from itertools import product, cycle
from operator import mul, add
from typing import TypeVar, Generic, Callable, Union, Dict, List, Any, overload, cast
def _process_keys(left: Cycler[K, V] | Iterable[dict[K, V]], right: Cycler[K, V] | Iterable[dict[K, V]] | None) -> set[K]:
    """
    Helper function to compose cycler keys.

    Parameters
    ----------
    left, right : iterable of dictionaries or None
        The cyclers to be composed.

    Returns
    -------
    keys : set
        The keys in the composition of the two cyclers.
    """
    l_peek: dict[K, V] = next(iter(left)) if left != [] else {}
    r_peek: dict[K, V] = next(iter(right)) if right is not None else {}
    l_key: set[K] = set(l_peek.keys())
    r_key: set[K] = set(r_peek.keys())
    if l_key & r_key:
        raise ValueError('Can not compose overlapping cycles')
    return l_key | r_key