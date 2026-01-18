from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage.util.dtype import img_as_float32
from concurrent.futures import ThreadPoolExecutor
def _texture_filter(gaussian_filtered):
    H_elems = [np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)]
    eigvals = feature.hessian_matrix_eigvals(H_elems)
    return eigvals