import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def _table_lookup(image, table):
    """
    Perform a morphological transform on an image, directed by its
    neighbors

    Parameters
    ----------
    image : ndarray
        A binary image
    table : ndarray
        A 512-element table giving the transform of each pixel given
        the values of that pixel and its 8-connected neighbors.

    Returns
    -------
    result : ndarray of same shape as `image`
        Transformed image

    Notes
    -----
    The pixels are numbered like this::

      0 1 2
      3 4 5
      6 7 8

    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true.
    """
    if image.shape[0] < 3 or image.shape[1] < 3:
        image = image.astype(bool)
        indexer = np.zeros(image.shape, int)
        indexer[1:, 1:] += image[:-1, :-1] * 2 ** 0
        indexer[1:, :] += image[:-1, :] * 2 ** 1
        indexer[1:, :-1] += image[:-1, 1:] * 2 ** 2
        indexer[:, 1:] += image[:, :-1] * 2 ** 3
        indexer[:, :] += image[:, :] * 2 ** 4
        indexer[:, :-1] += image[:, 1:] * 2 ** 5
        indexer[:-1, 1:] += image[1:, :-1] * 2 ** 6
        indexer[:-1, :] += image[1:, :] * 2 ** 7
        indexer[:-1, :-1] += image[1:, 1:] * 2 ** 8
    else:
        indexer = _table_lookup_index(np.ascontiguousarray(image, np.uint8))
    image = table[indexer]
    return image