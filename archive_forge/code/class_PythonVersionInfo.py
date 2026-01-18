import re
import sys
from ast import literal_eval
from functools import total_ordering
from typing import NamedTuple, Sequence, Union
@total_ordering
class PythonVersionInfo(_PythonVersionInfo):

    def __gt__(self, other):
        if isinstance(other, tuple):
            if len(other) != 2:
                raise ValueError('Can only compare to tuples of length 2.')
            return (self.major, self.minor) > other
        super().__gt__(other)
        return (self.major, self.minor)

    def __eq__(self, other):
        if isinstance(other, tuple):
            if len(other) != 2:
                raise ValueError('Can only compare to tuples of length 2.')
            return (self.major, self.minor) == other
        super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)