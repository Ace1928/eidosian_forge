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
def _pipe_segment_with_colons(align, colwidth):
    """Return a segment of a horizontal line with optional colons which
    indicate column's alignment (as in `pipe` output format)."""
    w = colwidth
    if align in ['right', 'decimal']:
        return '-' * (w - 1) + ':'
    elif align == 'center':
        return ':' + '-' * (w - 2) + ':'
    elif align == 'left':
        return ':' + '-' * (w - 1)
    else:
        return '-' * w