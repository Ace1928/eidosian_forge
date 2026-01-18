from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _format_multicolumn(self, row: list[str]) -> list[str]:
    """
        Combine columns belonging to a group to a single multicolumn entry
        according to self.multicolumn_format

        e.g.:
        a &  &  & b & c &
        will become
        \\multicolumn{3}{l}{a} & b & \\multicolumn{2}{l}{c}
        """
    row2 = row[:self.index_levels]
    ncol = 1
    coltext = ''

    def append_col() -> None:
        if ncol > 1:
            row2.append(f'\\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}{{{coltext.strip()}}}')
        else:
            row2.append(coltext)
    for c in row[self.index_levels:]:
        if c.strip():
            if coltext:
                append_col()
            coltext = c
            ncol = 1
        else:
            ncol += 1
    if coltext:
        append_col()
    return row2