import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class IsInstance:
    """Matcher that wraps isinstance."""

    def __init__(self, *types):
        self.types = tuple(types)

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join((type.__name__ for type in self.types)))

    def match(self, other):
        if isinstance(other, self.types):
            return None
        return NotAnInstance(other, self.types)