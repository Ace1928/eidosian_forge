import collections.abc
import copy
import pickle
from typing import (
def InvalidateIterators(self) -> None:
    original = self._values
    self._values = original.copy()
    original[None] = None