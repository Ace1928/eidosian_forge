import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
class SeparatedValues(str):
    """
    A string separated by a separator. Overrides __iter__ for getting
    the values.

    >>> list(SeparatedValues('a,b,c'))
    ['a', 'b', 'c']

    Whitespace is stripped and empty values are discarded.

    >>> list(SeparatedValues(' a,   b   , c,  '))
    ['a', 'b', 'c']
    """
    separator = ','

    def __iter__(self):
        parts = self.split(self.separator)
        return filter(None, (part.strip() for part in parts))