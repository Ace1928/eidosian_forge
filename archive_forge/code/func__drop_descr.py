import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _drop_descr(ndtype, drop_names):
    names = ndtype.names
    newdtype = []
    for name in names:
        current = ndtype[name]
        if name in drop_names:
            continue
        if current.names is not None:
            descr = _drop_descr(current, drop_names)
            if descr:
                newdtype.append((name, descr))
        else:
            newdtype.append((name, current))
    return newdtype