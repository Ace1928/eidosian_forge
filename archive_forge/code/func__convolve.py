from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
def _convolve(x, k):
    sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=True)
    return _do_convolve(x, k)[sx:ex, sy:ey]