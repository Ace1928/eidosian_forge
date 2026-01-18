import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def _pattern_of(index):
    """
    Return the pattern represented by an index value
    Byte decomposition of index
    """
    return np.array([[index & 2 ** 0, index & 2 ** 1, index & 2 ** 2], [index & 2 ** 3, index & 2 ** 4, index & 2 ** 5], [index & 2 ** 6, index & 2 ** 7, index & 2 ** 8]], bool)