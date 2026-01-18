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
def _build_row(padded_cells, colwidths, colaligns, rowfmt):
    """Return a string which represents a row of data cells."""
    if not rowfmt:
        return None
    if hasattr(rowfmt, '__call__'):
        return rowfmt(padded_cells, colwidths, colaligns)
    else:
        return _build_simple_row(padded_cells, rowfmt)