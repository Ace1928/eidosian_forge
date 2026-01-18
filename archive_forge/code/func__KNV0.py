import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def _KNV0(B, ker_pole, transfer_matrix, j, poles):
    """
    Algorithm "KNV0" Kautsky et Al. Robust pole
    assignment in linear state feedback, Int journal of Control
    1985, vol 41 p 1129->1155
    https://la.epfl.ch/files/content/sites/la/files/
        users/105941/public/KautskyNicholsDooren

    """
    transfer_matrix_not_j = cupy.delete(transfer_matrix, j, axis=1)
    Q, R = cupy.linalg.qr(transfer_matrix_not_j, mode='complete')
    mat_ker_pj = ker_pole[j] @ ker_pole[j].T
    yj = mat_ker_pj @ Q[:, -1]
    if not cupy.allclose(yj, 0):
        xj = yj / cupy.linalg.norm(yj)
        transfer_matrix[:, j] = xj