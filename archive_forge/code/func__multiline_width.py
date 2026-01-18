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
def _multiline_width(multiline_s, line_width_fn=len):
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split('[\r\n]', multiline_s)))