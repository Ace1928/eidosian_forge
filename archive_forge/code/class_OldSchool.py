from __future__ import annotations
from .entities import ComparableEntity
from ..schema import Column
from ..types import String
class OldSchool:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return other.__class__ is self.__class__ and other.x == self.x and (other.y == self.y)