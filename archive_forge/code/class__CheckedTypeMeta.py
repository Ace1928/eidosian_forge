from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class _CheckedTypeMeta(ABCMeta):

    def __new__(mcs, name, bases, dct):
        _store_types(dct, bases, '_checked_types', '__type__')
        store_invariants(dct, bases, '_checked_invariants', '__invariant__')

        def default_serializer(self, _, value):
            if isinstance(value, CheckedType):
                return value.serialize()
            return value
        dct.setdefault('__serializer__', default_serializer)
        dct['__slots__'] = ()
        return super(_CheckedTypeMeta, mcs).__new__(mcs, name, bases, dct)