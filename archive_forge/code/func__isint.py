from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _isint(string, inttype=int):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is inttype or (isinstance(string, (bytes, str)) and _isconvertible(inttype, string))