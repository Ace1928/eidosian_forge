import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _fix_defaults(output, defaults=None):
    """
    Update the fill_value and masked data of `output`
    from the default given in a dictionary defaults.
    """
    names = output.dtype.names
    data, mask, fill_value = (output.data, output.mask, output.fill_value)
    for k, v in (defaults or {}).items():
        if k in names:
            fill_value[k] = v
            data[k][mask[k]] = v
    return output