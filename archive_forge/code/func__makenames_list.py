import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _makenames_list(adict, align):
    allfields = []
    for fname, obj in adict.items():
        n = len(obj)
        if not isinstance(obj, tuple) or n not in (2, 3):
            raise ValueError('entry not a 2- or 3- tuple')
        if n > 2 and obj[2] == fname:
            continue
        num = int(obj[1])
        if num < 0:
            raise ValueError('invalid offset.')
        format = dtype(obj[0], align=align)
        if n > 2:
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]
    return (names, formats, offsets, titles)