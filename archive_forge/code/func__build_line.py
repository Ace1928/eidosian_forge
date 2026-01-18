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
def _build_line(colwidths, colaligns, linefmt):
    """Return a string which represents a horizontal line."""
    if not linefmt:
        return None
    if hasattr(linefmt, '__call__'):
        return linefmt(colwidths, colaligns)
    else:
        begin, fill, sep, end = linefmt
        cells = [fill * w for w in colwidths]
        return _build_simple_row(cells, (begin, sep, end))