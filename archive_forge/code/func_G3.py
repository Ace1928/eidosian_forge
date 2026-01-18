import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def G3(n):
    bits = nabe(n)
    return not ((bits[1] or bits[2] or (not bits[7])) and bits[0])