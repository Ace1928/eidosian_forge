import numpy as np
from scipy import ndimage as ndi
def _set_border_values(image, value, border_width=1):
    """Set edge values along all axes to a constant value.

    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.
    border_width : int or sequence of tuples
        A sequence with one 2-tuple per axis where the first and second values
        are the width of the border at the start and end of the axis,
        respectively. If an int is provided, a uniform border width along all
        axes is used.

    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_border_values(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    >>> image = np.zeros((8, 8), dtype=int)
    >>> _set_border_values(image, 1, border_width=((1, 1), (2, 3)))
    >>> image
    array([[1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    """
    if np.isscalar(border_width):
        border_width = ((border_width, border_width),) * image.ndim
    elif len(border_width) != image.ndim:
        raise ValueError('length of `border_width` must match image.ndim')
    for axis, npad in enumerate(border_width):
        if len(npad) != 2:
            raise ValueError('each sequence in `border_width` must have length 2')
        w_start, w_end = npad
        if w_start == w_end == 0:
            continue
        elif w_start == w_end == 1:
            sl = (slice(None),) * axis + ((0, -1),) + (...,)
            image[sl] = value
            continue
        if w_start > 0:
            sl = (slice(None),) * axis + (slice(0, w_start),) + (...,)
            image[sl] = value
        if w_end > 0:
            sl = (slice(None),) * axis + (slice(-w_end, None),) + (...,)
            image[sl] = value