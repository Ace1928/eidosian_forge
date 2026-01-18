import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _flags_fromnum(num):
    res = []
    for key in _flagnames:
        value = _flagdict[key]
        if num & value:
            res.append(key)
    return res