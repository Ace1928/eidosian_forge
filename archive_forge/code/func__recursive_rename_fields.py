import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _recursive_rename_fields(ndtype, namemapper):
    newdtype = []
    for name in ndtype.names:
        newname = namemapper.get(name, name)
        current = ndtype[name]
        if current.names is not None:
            newdtype.append((newname, _recursive_rename_fields(current, namemapper)))
        else:
            newdtype.append((newname, current))
    return newdtype