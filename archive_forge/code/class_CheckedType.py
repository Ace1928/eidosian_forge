from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class CheckedType(object):
    """
    Marker class to enable creation and serialization of checked object graphs.
    """
    __slots__ = ()

    @classmethod
    @abstractmethod
    def create(cls, source_data, _factory_fields=None):
        raise NotImplementedError()

    @abstractmethod
    def serialize(self, format=None):
        raise NotImplementedError()