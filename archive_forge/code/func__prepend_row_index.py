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
def _prepend_row_index(rows, index):
    """Add a left-most index column."""
    if index is None or index is False:
        return rows
    if isinstance(index, Sized) and len(index) != len(rows):
        raise ValueError('index must be as long as the number of data rows: ' + 'len(index)={} len(rows)={}'.format(len(index), len(rows)))
    sans_rows, separating_lines = _remove_separating_lines(rows)
    new_rows = []
    index_iter = iter(index)
    for row in sans_rows:
        index_v = next(index_iter)
        new_rows.append([index_v] + list(row))
    rows = new_rows
    _reinsert_separating_lines(rows, separating_lines)
    return rows