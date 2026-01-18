import math as _math
import time as _time
import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.api import runtime
@staticmethod
def _to_str_per_item(device_name, t):
    assert t.ndim == 1
    assert t.size > 0
    t_us = t * 1000000.0
    s = '    {}: {:9.03f} us'.format(device_name, t_us.mean())
    if t.size > 1:
        s += '   +/- {:6.03f} (min: {:9.03f} / max: {:9.03f}) us'.format(t_us.std(), t_us.min(), t_us.max())
    return s