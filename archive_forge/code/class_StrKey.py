from collections import abc
import itertools
from typing import (
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
class StrKey(str):
    """A string that can be compared to a string or sequence of strings representing a
    SeqStrType. This is needed for the tree functions to work.
    """

    def __lt__(self, other: SeqStrType):
        if isinstance(other, str):
            return str(self) < other
        else:
            return (self,) < tuple(other)

    def __gt__(self, other: SeqStrType):
        if isinstance(other, str):
            return str(self) > other
        else:
            return (self,) > tuple(other)