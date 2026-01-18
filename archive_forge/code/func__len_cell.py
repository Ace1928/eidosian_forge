from __future__ import division
import sys
import unicodedata
from functools import reduce
def _len_cell(self, cell):
    """Return the width of the cell

        Special characters are taken into account to return the width of the
        cell, such like newlines and tabs
        """
    cell_lines = cell.split('\n')
    maxi = 0
    for line in cell_lines:
        length = 0
        parts = line.split('\t')
        for part, i in zip(parts, list(range(1, len(parts) + 1))):
            length = length + len(part)
            if i < len(parts):
                length = (length // 8 + 1) * 8
        maxi = max(maxi, length)
    return maxi