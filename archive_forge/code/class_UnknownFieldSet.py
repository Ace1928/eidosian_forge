import collections.abc
import copy
import pickle
from typing import (
class UnknownFieldSet:
    """UnknownField container"""
    __slots__ = ['_values']

    def __init__(self):
        self._values = []

    def __getitem__(self, index):
        if self._values is None:
            raise ValueError('UnknownFields does not exist. The parent message might be cleared.')
        size = len(self._values)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError('index %d out of range'.index)
        return UnknownFieldRef(self, index)

    def _internal_get(self, index):
        return self._values[index]

    def __len__(self):
        if self._values is None:
            raise ValueError('UnknownFields does not exist. The parent message might be cleared.')
        return len(self._values)

    def _add(self, field_number, wire_type, data):
        unknown_field = _UnknownField(field_number, wire_type, data)
        self._values.append(unknown_field)
        return unknown_field

    def __iter__(self):
        for i in range(len(self)):
            yield UnknownFieldRef(self, i)

    def _extend(self, other):
        if other is None:
            return
        self._values.extend(other._values)

    def __eq__(self, other):
        if self is other:
            return True
        values = list(self._values)
        if other is None:
            return not values
        values.sort()
        other_values = sorted(other._values)
        return values == other_values

    def _clear(self):
        for value in self._values:
            if isinstance(value._data, UnknownFieldSet):
                value._data._clear()
        self._values = None