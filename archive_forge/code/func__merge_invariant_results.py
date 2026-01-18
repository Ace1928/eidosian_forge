from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def _merge_invariant_results(result):
    verdict = True
    data = []
    for verd, dat in result:
        if not verd:
            verdict = False
            data.append(dat)
    return (verdict, tuple(data))