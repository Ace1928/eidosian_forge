import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def G3p(n):
    bits = nabe(n)
    return not ((bits[5] or bits[6] or (not bits[3])) and bits[4])