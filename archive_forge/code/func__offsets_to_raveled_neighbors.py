import numpy as np
from scipy import ndimage as ndi
def _offsets_to_raveled_neighbors(image_shape, footprint, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.

    Parameters
    ----------
    image_shape : tuple
        The shape of the image for which the offsets are computed.
    footprint : ndarray
        The footprint (structuring element) determining the neighborhood
        expressed as an n-D array of 1's and 0's.
    center : tuple
        Tuple of indices to the center of `footprint`.
    order : {"C", "F"}, optional
        Whether the image described by `image_shape` is in row-major (C-style)
        or column-major (Fortran-style) order.

    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.

    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.

    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> _offsets_to_raveled_neighbors((2, 3, 2), np.ones((3, 3, 3)), (1, 1, 1))
    array([-6, -2, -1,  1,  2,  6, -8, -7, -5, -4, -3,  3,  4,  5,  7,  8, -9,
            9])
    """
    raveled_offsets = _raveled_offsets_and_distances(image_shape, footprint=footprint, center=center, order=order)[0]
    return raveled_offsets