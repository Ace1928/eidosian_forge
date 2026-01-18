import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def _read_valuation_line(s):
    """
    Read a line in a valuation file.

    Lines are expected to be of the form::

      noosa => n
      girl => {g1, g2}
      chase => {(b1, g1), (b2, g1), (g1, d1), (g2, d2)}

    :param s: input line
    :type s: str
    :return: a pair (symbol, value)
    :rtype: tuple
    """
    pieces = _VAL_SPLIT_RE.split(s)
    symbol = pieces[0]
    value = pieces[1]
    if value.startswith('{'):
        value = value[1:-1]
        tuple_strings = _TUPLES_RE.findall(value)
        if tuple_strings:
            set_elements = []
            for ts in tuple_strings:
                ts = ts[1:-1]
                element = tuple(_ELEMENT_SPLIT_RE.split(ts))
                set_elements.append(element)
        else:
            set_elements = _ELEMENT_SPLIT_RE.split(value)
        value = set(set_elements)
    return (symbol, value)