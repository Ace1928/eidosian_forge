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
def _wrap_text_to_colwidths(list_of_lists, colwidths, numparses=True):
    numparses = _expand_iterable(numparses, len(list_of_lists[0]), True)
    result = []
    for row in list_of_lists:
        new_row = []
        for cell, width, numparse in zip(row, colwidths, numparses):
            if _isnumber(cell) and numparse:
                new_row.append(cell)
                continue
            if width is not None:
                wrapper = _CustomTextWrap(width=width)
                casted_cell = str(cell) if _isnumber(cell) else _type(cell, numparse)(cell)
                wrapped = wrapper.wrap(casted_cell)
                new_row.append('\n'.join(wrapped))
            else:
                new_row.append(cell)
        result.append(new_row)
    return result