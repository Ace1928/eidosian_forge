from scipy._lib._array_api import (
from . import _pocketfft
import numpy as np
def _validate_fft_args(workers, plan, norm):
    if workers is not None:
        raise ValueError(xp_unsupported_param_msg('workers'))
    if plan is not None:
        raise ValueError(xp_unsupported_param_msg('plan'))
    if norm is None:
        norm = 'backward'
    return norm