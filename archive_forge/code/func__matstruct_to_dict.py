import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
def _matstruct_to_dict(matobj):
    """Construct nested dicts from mat_struct objects."""
    d = {}
    for f in matobj._fieldnames:
        elem = matobj.__dict__[f]
        if isinstance(elem, mat_struct):
            d[f] = _matstruct_to_dict(elem)
        elif _has_struct(elem):
            d[f] = _inspect_cell_array(elem)
        else:
            d[f] = elem
    return d