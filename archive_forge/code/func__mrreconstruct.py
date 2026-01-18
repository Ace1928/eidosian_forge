from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def _mrreconstruct(subtype, baseclass, baseshape, basetype):
    """
    Build a new MaskedArray from the information stored in a pickle.

    """
    _data = ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
    _mask = ndarray.__new__(ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype)