import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _get_decode3d_code(size_max, int_type=''):
    if size_max > 1024:
        code = f'\n        {int_type} x = (encoded >> 40) & 0xfffff;\n        {int_type} y = (encoded >> 20) & 0xfffff;\n        {int_type} z = encoded & 0xfffff;\n        '
    else:
        code = f'\n        {int_type} x = (encoded >> 20) & 0x3ff;\n        {int_type} y = (encoded >> 10) & 0x3ff;\n        {int_type} z = encoded & 0x3ff;\n        '
    return code