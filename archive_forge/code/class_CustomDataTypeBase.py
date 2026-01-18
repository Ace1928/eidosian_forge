import unittest
from zope.interface.tests import OptimizationTestMixin
class CustomDataTypeBase:
    _data = None

    def __getitem__(self, name):
        return self._data[name]

    def __setitem__(self, name, value):
        self._data[name] = value

    def __delitem__(self, name):
        del self._data[name]

    def __len__(self):
        return len(self._data)

    def __contains__(self, name):
        return name in self._data

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) != type(self):
            return False
        return other._data == self._data

    def __repr__(self):
        return repr(self._data)