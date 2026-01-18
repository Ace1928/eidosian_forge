import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _strcoll(a, b):
    """ strcoll(string,string) -> int.
        Compares two strings according to the locale.
    """
    return (a > b) - (a < b)