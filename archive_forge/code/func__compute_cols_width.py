from __future__ import division
import sys
import unicodedata
from functools import reduce
def _compute_cols_width(self):
    """Return an array with the width of each column

        If a specific width has been specified, exit. If the total of the
        columns width exceed the table desired width, another width will be
        computed to fit, and cells will be wrapped.
        """
    if hasattr(self, '_width'):
        return
    maxi = []
    if self._header:
        maxi = [self._len_cell(x) for x in self._header]
    for row in self._rows:
        for cell, i in zip(row, list(range(len(row)))):
            try:
                maxi[i] = max(maxi[i], self._len_cell(cell))
            except (TypeError, IndexError):
                maxi.append(self._len_cell(cell))
    ncols = len(maxi)
    content_width = sum(maxi)
    deco_width = 3 * (ncols - 1) + [0, 4][self._has_border()]
    if self._max_width and content_width + deco_width > self._max_width:
        " content too wide to fit the expected max_width\n            let's recompute maximum cell width for each cell\n            "
        if self._max_width < ncols + deco_width:
            raise ValueError('max_width too low to render data')
        available_width = self._max_width - deco_width
        newmaxi = [0] * ncols
        i = 0
        while available_width > 0:
            if newmaxi[i] < maxi[i]:
                newmaxi[i] += 1
                available_width -= 1
            i = (i + 1) % ncols
        maxi = newmaxi
    self._width = maxi