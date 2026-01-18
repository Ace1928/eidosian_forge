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
def _handle_truncated_float_vec(vec, nbytes):
    if nbytes != 8:
        vec1 = np.zeros(len(vec), np.dtype('S8'))
        dtype = np.dtype(f'S{nbytes},S{8 - nbytes}')
        vec2 = vec1.view(dtype=dtype)
        vec2['f0'] = vec
        return vec2
    return vec