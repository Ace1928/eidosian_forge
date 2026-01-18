import numpy as np
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
import functools
def _r2rn(forward, transform, x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """Forward or backward nd DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    if len(axes) == 0:
        return x
    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied
    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2
    norm = _normalization(norm, forward)
    workers = _workers(workers)
    out = tmp if overwrite_x else None
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out
        transform(tmp.real, type, axes, norm, out.real, workers)
        transform(tmp.imag, type, axes, norm, out.imag, workers)
        return out
    return transform(tmp, type, axes, norm, out, workers, orthogonalize)