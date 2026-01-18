from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
def _join_multiline(self, strcols_input: Iterable[list[str]]) -> str:
    lwidth = self.line_width
    adjoin_width = 1
    strcols = list(strcols_input)
    if self.fmt.index:
        idx = strcols.pop(0)
        lwidth -= np.array([self.adj.len(x) for x in idx]).max() + adjoin_width
    col_widths = [np.array([self.adj.len(x) for x in col]).max() if len(col) > 0 else 0 for col in strcols]
    assert lwidth is not None
    col_bins = _binify(col_widths, lwidth)
    nbins = len(col_bins)
    str_lst = []
    start = 0
    for i, end in enumerate(col_bins):
        row = strcols[start:end]
        if self.fmt.index:
            row.insert(0, idx)
        if nbins > 1:
            nrows = len(row[-1])
            if end <= len(strcols) and i < nbins - 1:
                row.append([' \\'] + ['  '] * (nrows - 1))
            else:
                row.append([' '] * nrows)
        str_lst.append(self.adj.adjoin(adjoin_width, *row))
        start = end
    return '\n\n'.join(str_lst)