from __future__ import annotations
from .entities import ComparableEntity
from ..schema import Column
from ..types import String
class OldSchoolWithoutCompare:

    def __init__(self, x, y):
        self.x = x
        self.y = y