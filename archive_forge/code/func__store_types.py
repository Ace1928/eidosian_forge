from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def _store_types(dct, bases, destination_name, source_name):
    maybe_types = maybe_parse_many_user_types([d[source_name] for d in [dct] + [b.__dict__ for b in bases] if source_name in d])
    dct[destination_name] = maybe_types