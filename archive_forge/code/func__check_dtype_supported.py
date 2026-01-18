import numpy as np
import functools
from scipy import ndimage as ndi
from .._shared.utils import warn
def _check_dtype_supported(ar):
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError(f'Only bool or integer image types are supported. Got {ar.dtype}.')