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
def _expand_numparse(disable_numparse, column_count):
    """
    Return a list of bools of length `column_count` which indicates whether
    number parsing should be used on each column.
    If `disable_numparse` is a list of indices, each of those indices are False,
    and everything else is True.
    If `disable_numparse` is a bool, then the returned list is all the same.
    """
    if isinstance(disable_numparse, Iterable):
        numparses = [True] * column_count
        for index in disable_numparse:
            numparses[index] = False
        return numparses
    else:
        return [not disable_numparse] * column_count