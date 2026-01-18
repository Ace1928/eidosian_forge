import collections.abc
import copy
import pickle
from typing import (
def MergeFrom(self, other: 'MessageMap[_K, _V]') -> None:
    for key in other._values:
        if key in self:
            del self[key]
        self[key].CopyFrom(other[key])