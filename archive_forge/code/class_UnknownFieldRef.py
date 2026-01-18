import collections.abc
import copy
import pickle
from typing import (
class UnknownFieldRef:

    def __init__(self, parent, index):
        self._parent = parent
        self._index = index

    def _check_valid(self):
        if not self._parent:
            raise ValueError('UnknownField does not exist. The parent message might be cleared.')
        if self._index >= len(self._parent):
            raise ValueError('UnknownField does not exist. The parent message might be cleared.')

    @property
    def field_number(self):
        self._check_valid()
        return self._parent._internal_get(self._index)._field_number

    @property
    def wire_type(self):
        self._check_valid()
        return self._parent._internal_get(self._index)._wire_type

    @property
    def data(self):
        self._check_valid()
        return self._parent._internal_get(self._index)._data