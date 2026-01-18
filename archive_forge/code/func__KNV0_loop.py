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
def _KNV0_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Loop over all poles one by one and apply KNV method 0 algorithm
    """
    stop = False
    nb_try = 0
    while nb_try < maxiter and (not stop):
        det_transfer_matrixb = cupy.abs(cupy.linalg.det(transfer_matrix))
        for j in range(B.shape[0]):
            _KNV0(B, ker_pole, transfer_matrix, j, poles)
        sq_spacing = sqrt(sqrt(cupy.finfo(cupy.float64).eps))
        det_transfer_matrix = max((sq_spacing, cupy.abs(cupy.linalg.det(transfer_matrix))))
        cur_rtol = cupy.abs((det_transfer_matrix - det_transfer_matrixb) / det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > sq_spacing:
            stop = True
        nb_try += 1
    return (stop, cur_rtol, nb_try)