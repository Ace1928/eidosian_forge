from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class _CheckedMapTypeMeta(type):

    def __new__(mcs, name, bases, dct):
        _store_types(dct, bases, '_checked_key_types', '__key_type__')
        _store_types(dct, bases, '_checked_value_types', '__value_type__')
        store_invariants(dct, bases, '_checked_invariants', '__invariant__')

        def default_serializer(self, _, key, value):
            sk = key
            if isinstance(key, CheckedType):
                sk = key.serialize()
            sv = value
            if isinstance(value, CheckedType):
                sv = value.serialize()
            return (sk, sv)
        dct.setdefault('__serializer__', default_serializer)
        dct['__slots__'] = ()
        return super(_CheckedMapTypeMeta, mcs).__new__(mcs, name, bases, dct)