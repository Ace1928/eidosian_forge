from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
class TheMap(CheckedPMap):
    __key_type__ = key_type
    __value_type__ = value_type

    def __reduce__(self):
        return (_restore_pmap_field_pickle, (self.__key_type__, self.__value_type__, dict(self)))