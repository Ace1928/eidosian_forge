from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
class MatchVariable(object):
    """
    Represents a match of a variable in the grammar.

    :param varname: (string) Name of the variable.
    :param value: (string) Value of this variable.
    :param slice: (start, stop) tuple, indicating the position of this variable
                  in the input string.
    """

    def __init__(self, varname, value, slice):
        self.varname = varname
        self.value = value
        self.slice = slice
        self.start = self.slice[0]
        self.stop = self.slice[1]

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.varname, self.value)