import numpy as np
from scipy import ndimage as ndi
def _resolve_neighborhood(footprint, connectivity, ndim, enforce_adjacency=True):
    """Validate or create a footprint (structuring element).

    Depending on the values of `connectivity` and `footprint` this function
    either creates a new footprint (`footprint` is None) using `connectivity`
    or validates the given footprint (`footprint` is not None).

    Parameters
    ----------
    footprint : ndarray
        The footprint (structuring) element used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    ndim : int
        Number of dimensions `footprint` ought to have.
    enforce_adjacency : bool
        A boolean that determines whether footprint must only specify direct
        neighbors.

    Returns
    -------
    footprint : ndarray
        Validated or new footprint specifying the neighborhood.

    Examples
    --------
    >>> _resolve_neighborhood(None, 1, 2)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]])
    >>> _resolve_neighborhood(None, None, 3).shape
    (3, 3, 3)
    """
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        footprint = ndi.generate_binary_structure(ndim, connectivity)
    else:
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.ndim != ndim:
            raise ValueError('number of dimensions in image and footprint do notmatch')
        if enforce_adjacency and any((s != 3 for s in footprint.shape)):
            raise ValueError('dimension size in footprint is not 3')
        elif any((s % 2 != 1 for s in footprint.shape)):
            raise ValueError('footprint size must be odd along all dimensions')
    return footprint