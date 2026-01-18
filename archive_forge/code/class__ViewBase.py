import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class _ViewBase:

    def __init__(self, impl):
        self._impl = impl

    def __len__(self):
        return len(self._impl._items)