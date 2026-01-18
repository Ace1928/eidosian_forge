import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner
def _get_neigh_coef(shape, center, dtype=float):
    neigh_coef = np.zeros(shape, dtype=dtype)
    neigh_coef[center] = 1
    neigh_coef = laplace(laplace(neigh_coef))
    coef_idx = np.where(neigh_coef)
    coef_vals = neigh_coef[coef_idx]
    coef_idx = np.stack(coef_idx, axis=0)
    return (neigh_coef, coef_idx, coef_vals)