from __future__ import annotations
from .entities import ComparableEntity
from ..schema import Column
from ..types import String
class BrokenComparable:

    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError