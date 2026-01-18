import types
from ._impl import (
class PostfixedMismatch(MismatchDecorator):
    """A mismatch annotated with a descriptive string."""

    def __init__(self, annotation, mismatch):
        super().__init__(mismatch)
        self.annotation = annotation
        self.mismatch = mismatch

    def describe(self):
        return f'{self.original.describe()}: {self.annotation}'