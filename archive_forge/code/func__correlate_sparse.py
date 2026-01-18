import numpy as np
from .._shared.utils import _supported_float_type, _to_np_mode
def _correlate_sparse(image, kernel_shape, kernel_indices, kernel_values):
    """Perform correlation with a sparse kernel.

    Parameters
    ----------
    image : ndarray
        The (prepadded) image to be correlated.
    kernel_shape : tuple of int
        The shape of the sparse filter kernel.
    kernel_indices : list of coordinate tuples
        The indices of each non-zero kernel entry.
    kernel_values : list of float
        The kernel values at each location in kernel_indices.

    Returns
    -------
    out : ndarray
        The filtered image.

    Notes
    -----
    This function only returns results for the 'valid' region of the
    convolution, and thus `out` will be smaller than `image` by an amount
    equal to the kernel size along each axis.
    """
    idx, val = (kernel_indices[0], kernel_values[0])
    if tuple(idx) != (0,) * image.ndim:
        raise RuntimeError('Unexpected initial index in kernel_indices')
    out = _get_view(image, kernel_shape, idx, val).copy()
    for idx, val in zip(kernel_indices[1:], kernel_values[1:]):
        out += _get_view(image, kernel_shape, idx, val)
    return out