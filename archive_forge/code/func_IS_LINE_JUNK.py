from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def IS_LINE_JUNK(line, pat=re.compile('\\s*(?:#\\s*)?$').match):
    """
    Return True for ignorable line: iff `line` is blank or contains a single '#'.

    Examples:

    >>> IS_LINE_JUNK('\\n')
    True
    >>> IS_LINE_JUNK('  #   \\n')
    True
    >>> IS_LINE_JUNK('hello\\n')
    False
    """
    return pat(line) is not None