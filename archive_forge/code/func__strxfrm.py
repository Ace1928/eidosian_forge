import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _strxfrm(s):
    """ strxfrm(string) -> string.
        Returns a string that behaves for cmp locale-aware.
    """
    return s