from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def _all_dicts(bases, seen=None):
    """
    Yield each class in ``bases`` and each of their base classes.
    """
    if seen is None:
        seen = set()
    for cls in bases:
        if cls in seen:
            continue
        seen.add(cls)
        yield cls.__dict__
        for b in _all_dicts(cls.__bases__, seen):
            yield b