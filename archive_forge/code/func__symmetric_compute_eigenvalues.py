import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def _symmetric_compute_eigenvalues(S_elems):
    """Compute eigenvalues from the upper-diagonal entries of a symmetric
    matrix.

    Parameters
    ----------
    S_elems : list of ndarray
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the matrix, in decreasing order. The eigenvalues are
        the leading dimension. That is, ``eigs[i, j, k]`` contains the
        ith-largest eigenvalue at position (j, k).
    """
    if len(S_elems) == 3:
        M00, M01, M11 = S_elems
        eigs = np.empty((2, *M00.shape), M00.dtype)
        eigs[:] = (M00 + M11) / 2
        hsqrtdet = np.sqrt(M01 ** 2 + ((M00 - M11) / 2) ** 2)
        eigs[0] += hsqrtdet
        eigs[1] -= hsqrtdet
        return eigs
    else:
        matrices = _symmetric_image(S_elems)
        eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigs.ndim - 1))
        return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)