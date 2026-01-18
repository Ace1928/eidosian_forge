from Markus Kuhn's C code, retrieved from:
from __future__ import division
import os
import sys
import warnings
from .table_wide import WIDE_EASTASIAN
from .table_zero import ZERO_WIDTH
from .unicode_versions import list_versions
def _bisearch(ucs, table):
    """
    Auxiliary function for binary search in interval table.

    :arg int ucs: Ordinal value of unicode character.
    :arg list table: List of starting and ending ranges of ordinal values,
        in form of ``[(start, end), ...]``.
    :rtype: int
    :returns: 1 if ordinal value ucs is found within lookup table, else 0.
    """
    lbound = 0
    ubound = len(table) - 1
    if ucs < table[0][0] or ucs > table[ubound][1]:
        return 0
    while ubound >= lbound:
        mid = (lbound + ubound) // 2
        if ucs > table[mid][1]:
            lbound = mid + 1
        elif ucs < table[mid][0]:
            ubound = mid - 1
        else:
            return 1
    return 0