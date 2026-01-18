from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
def _binify(cols: list[int], line_width: int) -> list[int]:
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last_column = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last_column == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined
    bins.append(len(cols))
    return bins