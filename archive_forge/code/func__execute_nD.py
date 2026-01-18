from scipy._lib._array_api import (
from . import _pocketfft
import numpy as np
def _execute_nD(func_str, pocketfft_func, x, s, axes, norm, overwrite_x, workers, plan):
    xp = array_namespace(x)
    if is_numpy(xp):
        return pocketfft_func(x, s=s, axes=axes, norm=norm, overwrite_x=overwrite_x, workers=workers, plan=plan)
    norm = _validate_fft_args(workers, plan, norm)
    if hasattr(xp, 'fft'):
        xp_func = getattr(xp.fft, func_str)
        return xp_func(x, s=s, axes=axes, norm=norm)
    x = np.asarray(x)
    y = pocketfft_func(x, s=s, axes=axes, norm=norm)
    return xp.asarray(y)