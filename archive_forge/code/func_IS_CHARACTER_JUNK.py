from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def IS_CHARACTER_JUNK(ch, ws=' \t'):
    """
    Return True for ignorable character: iff `ch` is a space or tab.

    Examples:

    >>> IS_CHARACTER_JUNK(' ')
    True
    >>> IS_CHARACTER_JUNK('\\t')
    True
    >>> IS_CHARACTER_JUNK('\\n')
    False
    >>> IS_CHARACTER_JUNK('x')
    False
    """
    return ch in ws