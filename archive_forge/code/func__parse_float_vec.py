from __future__ import annotations
from collections import abc
from datetime import datetime
import struct
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas.io.common import get_handle
from pandas.io.sas.sasreader import ReaderBase
def _parse_float_vec(vec):
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """
    dtype = np.dtype('>u4,>u4')
    vec1 = vec.view(dtype=dtype)
    xport1 = vec1['f0']
    xport2 = vec1['f1']
    ieee1 = xport1 & 16777215
    shift = np.zeros(len(vec), dtype=np.uint8)
    shift[np.where(xport1 & 2097152)] = 1
    shift[np.where(xport1 & 4194304)] = 2
    shift[np.where(xport1 & 8388608)] = 3
    ieee1 >>= shift
    ieee2 = xport2 >> shift | (xport1 & 7) << 29 + (3 - shift)
    ieee1 &= 4293918719
    ieee1 |= ((xport1 >> 24 & 127) - 65 << 2) + shift + 1023 << 20 | xport1 & 2147483648
    ieee = np.empty((len(ieee1),), dtype='>u4,>u4')
    ieee['f0'] = ieee1
    ieee['f1'] = ieee2
    ieee = ieee.view(dtype='>f8')
    ieee = ieee.astype('f8')
    return ieee