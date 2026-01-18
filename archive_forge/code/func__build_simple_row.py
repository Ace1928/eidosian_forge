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
def _build_simple_row(padded_cells, rowfmt):
    """Format row according to DataRow format without padding."""
    begin, sep, end = rowfmt
    return (begin + sep.join(padded_cells) + end).rstrip()