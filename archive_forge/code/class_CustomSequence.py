import unittest
from zope.interface.tests import OptimizationTestMixin
class CustomSequence(CustomDataTypeBase):

    def __init__(self, other=None):
        self._data = []
        if other:
            self._data.extend(other)
        self.append = self._data.append