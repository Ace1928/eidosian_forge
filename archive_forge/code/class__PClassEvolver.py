from pyrsistent._checked_types import (InvariantException, CheckedType, _restore_pickle, store_invariants)
from pyrsistent._field_common import (
from pyrsistent._transformations import transform
class _PClassEvolver(object):
    __slots__ = ('_pclass_evolver_original', '_pclass_evolver_data', '_pclass_evolver_data_is_dirty', '_factory_fields')

    def __init__(self, original, initial_dict):
        self._pclass_evolver_original = original
        self._pclass_evolver_data = initial_dict
        self._pclass_evolver_data_is_dirty = False
        self._factory_fields = set()

    def __getitem__(self, item):
        return self._pclass_evolver_data[item]

    def set(self, key, value):
        if self._pclass_evolver_data.get(key, _MISSING_VALUE) is not value:
            self._pclass_evolver_data[key] = value
            self._factory_fields.add(key)
            self._pclass_evolver_data_is_dirty = True
        return self

    def __setitem__(self, key, value):
        self.set(key, value)

    def remove(self, item):
        if item in self._pclass_evolver_data:
            del self._pclass_evolver_data[item]
            self._factory_fields.discard(item)
            self._pclass_evolver_data_is_dirty = True
            return self
        raise AttributeError(item)

    def __delitem__(self, item):
        self.remove(item)

    def persistent(self):
        if self._pclass_evolver_data_is_dirty:
            return self._pclass_evolver_original.__class__(_factory_fields=self._factory_fields, **self._pclass_evolver_data)
        return self._pclass_evolver_original

    def __setattr__(self, key, value):
        if key not in self.__slots__:
            self.set(key, value)
        else:
            super(_PClassEvolver, self).__setattr__(key, value)

    def __getattr__(self, item):
        return self[item]