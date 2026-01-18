import numpy as np
from .._shared.utils import _supported_float_type, _to_np_mode
def _validate_window_size(axis_sizes):
    """Ensure all sizes in ``axis_sizes`` are odd.

    Parameters
    ----------
    axis_sizes : iterable of int

    Raises
    ------
    ValueError
        If any given axis size is even.
    """
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = f'Window size for `threshold_sauvola` or `threshold_niblack` must not be even on any dimension. Got {axis_sizes}'
            raise ValueError(msg)