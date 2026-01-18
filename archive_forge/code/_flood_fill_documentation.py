import numpy as np
from ..util import crop
from ._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance
from ._util import (
from .._shared.dtype import numeric_dtype_min_max
Mask corresponding to a flood fill.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`. If not given, all adjacent pixels
        are considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and if within
        tolerance of the initial value will also be filled (inclusive).

    Returns
    -------
    mask : ndarray
        A Boolean array with the same shape as `image` is returned, with True
        values for areas connected to and equal (or within tolerance of) the
        seed point.  All other values are False.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.  This function returns just the mask
    representing the fill.

    If indices are desired rather than masks for memory reasons, the user can
    simply run `numpy.nonzero` on the result, save the indices, and discard
    this mask.

    Examples
    --------
    >>> from skimage.morphology import flood
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> mask = flood(image, (1, 1))
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> mask = flood(image, (1, 1), connectivity=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> mask = flood(image, (0, 0), tolerance=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    