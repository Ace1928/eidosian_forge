from __future__ import annotations
from collections.abc import Hashable, Iterable, Generator
import copy
from functools import reduce
from itertools import product, cycle
from operator import mul, add
from typing import TypeVar, Generic, Callable, Union, Dict, List, Any, overload, cast
@classmethod
def _from_iter(cls, label: K, itr: Iterable[V]) -> Cycler[K, V]:
    """
        Class method to create 'base' Cycler objects
        that do not have a 'right' or 'op' and for which
        the 'left' object is not another Cycler.

        Parameters
        ----------
        label : hashable
            The property key.

        itr : iterable
            Finite length iterable of the property values.

        Returns
        -------
        `Cycler`
            New 'base' cycler.
        """
    ret: Cycler[K, V] = cls(None)
    ret._left = list(({label: v} for v in itr))
    ret._keys = {label}
    return ret