from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
class SimpleSizeof:
    """Sentinel class to mark a class to be skipped by the dispatcher. This only
    works if this sentinel mixin is first in the mro.

    Examples
    --------
    >>> def _get_gc_overhead():
    ...     class _CustomObject:
    ...         def __sizeof__(self):
    ...             return 0
    ...
    ...     return sys.getsizeof(_CustomObject())

    >>> class TheAnswer(SimpleSizeof):
    ...     def __sizeof__(self):
    ...         # Sizeof always add overhead of an object for GC
    ...         return 42 - _get_gc_overhead()

    >>> sizeof(TheAnswer())
    42

    """