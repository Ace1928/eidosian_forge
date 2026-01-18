from itertools import product
import numpy as np
from .._shared import utils
from .._shared.utils import warn
def _run_one_shift(shift):
    xs = np.roll(x, shift, axis=roll_axes)
    tmp = func(xs, **func_kw)
    return np.roll(tmp, tuple((-s for s in shift)), axis=roll_axes)