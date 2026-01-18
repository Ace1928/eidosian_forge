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
def _moin_row_with_attrs(celltag, cell_values, colwidths, colaligns, header=''):
    alignment = {'left': '', 'right': '<style="text-align: right;">', 'center': '<style="text-align: center;">', 'decimal': '<style="text-align: right;">'}
    values_with_attrs = ['{}{} {} '.format(celltag, alignment.get(a, ''), header + c + header) for c, a in zip(cell_values, colaligns)]
    return ''.join(values_with_attrs) + '||'