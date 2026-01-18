import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class _ValuesView(_ViewBase, abc.ValuesView):

    def __contains__(self, value):
        for item in self._impl._items:
            if item[2] == value:
                return True
        return False

    def __iter__(self):
        return _Iter(len(self), self._iter(self._impl._version))

    def _iter(self, version):
        for item in self._impl._items:
            if version != self._impl._version:
                raise RuntimeError('Dictionary changed during iteration')
            yield item[2]

    def __repr__(self):
        lst = []
        for item in self._impl._items:
            lst.append('{!r}'.format(item[2]))
        body = ', '.join(lst)
        return '{}({})'.format(self.__class__.__name__, body)