from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
def _normalization(norm, forward):
    """Returns the pypocketfft normalization mode from the norm argument"""
    try:
        inorm = _NORM_MAP[norm]
        return inorm if forward else 2 - inorm
    except KeyError:
        raise ValueError(f'Invalid norm value {norm!r}, should be "backward", "ortho" or "forward"') from None