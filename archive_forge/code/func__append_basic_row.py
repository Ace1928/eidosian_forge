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
def _append_basic_row(lines, padded_cells, colwidths, colaligns, rowfmt, rowalign=None):
    lines.append(_build_row(padded_cells, colwidths, colaligns, rowfmt))
    return lines