from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
class _PField(object):
    __slots__ = ('type', 'invariant', 'initial', 'mandatory', '_factory', 'serializer')

    def __init__(self, type, invariant, initial, mandatory, factory, serializer):
        self.type = type
        self.invariant = invariant
        self.initial = initial
        self.mandatory = mandatory
        self._factory = factory
        self.serializer = serializer

    @property
    def factory(self):
        if self._factory is PFIELD_NO_FACTORY and len(self.type) == 1:
            typ = get_type(tuple(self.type)[0])
            if issubclass(typ, CheckedType):
                return typ.create
        return self._factory