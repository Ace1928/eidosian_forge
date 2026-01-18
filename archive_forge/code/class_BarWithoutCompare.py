from __future__ import annotations
from .entities import ComparableEntity
from ..schema import Column
from ..types import String
class BarWithoutCompare:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'Bar(%d, %d)' % (self.x, self.y)