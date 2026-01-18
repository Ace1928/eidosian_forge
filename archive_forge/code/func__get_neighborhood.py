import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner
def _get_neighborhood(nd_idx, radius, nd_shape):
    bounds_lo = np.maximum(nd_idx - radius, 0)
    bounds_hi = np.minimum(nd_idx + radius + 1, nd_shape)
    return (bounds_lo, bounds_hi)