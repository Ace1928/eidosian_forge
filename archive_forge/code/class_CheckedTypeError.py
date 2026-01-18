from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class CheckedTypeError(TypeError):

    def __init__(self, source_class, expected_types, actual_type, actual_value, *args, **kwargs):
        super(CheckedTypeError, self).__init__(*args, **kwargs)
        self.source_class = source_class
        self.expected_types = expected_types
        self.actual_type = actual_type
        self.actual_value = actual_value