import types
from ._impl import (
class PrefixedMismatch(MismatchDecorator):

    def __init__(self, prefix, mismatch):
        super().__init__(mismatch)
        self.prefix = prefix

    def describe(self):
        return f'{self.prefix}: {self.original.describe()}'