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
def _mediawiki_row_with_attrs(separator, cell_values, colwidths, colaligns):
    alignment = {'left': '', 'right': 'align="right"| ', 'center': 'align="center"| ', 'decimal': 'align="right"| '}
    values_with_attrs = [' ' + alignment.get(a, '') + c + ' ' for c, a in zip(cell_values, colaligns)]
    colsep = separator * 2
    return (separator + colsep.join(values_with_attrs)).rstrip()