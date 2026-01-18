from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
class FixedSaySomething:
    __slots__ = ('phrase',)

    def __init__(self, phrase):
        self.phrase = phrase