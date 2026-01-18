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
def _YT_complex(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.2 page 20 related to complex pairs
    """
    ur = sqrt(2) * Q[:, -2, None]
    ui = sqrt(2) * Q[:, -1, None]
    u = ur + 1j * ui
    ker_pole_ij = ker_pole[i]
    m = ker_pole_ij.conj().T @ (u @ u.conj().T - u.conj() @ u.T) @ ker_pole_ij
    import numpy as np
    e_val, e_vec = np.linalg.eig(m.get())
    e_val, e_vec = (cupy.asarray(e_val), cupy.asarray(e_vec))
    e_val_idx = cupy.argsort(cupy.abs(e_val))
    mu1 = e_vec[:, e_val_idx[-1], None]
    mu2 = e_vec[:, e_val_idx[-2], None]
    transfer_matrix_j_mo_transfer_matrix_j = transfer_matrix[:, i, None] + 1j * transfer_matrix[:, j, None]
    if not cupy.allclose(cupy.abs(e_val[e_val_idx[-1]]), cupy.abs(e_val[e_val_idx[-2]])):
        ker_pole_mu = ker_pole_ij @ mu1
    else:
        mu1_mu2_matrix = cupy.hstack((mu1, mu2))
        ker_pole_mu = ker_pole_ij @ mu1_mu2_matrix
    transfer_matrix_i_j = cupy.dot(ker_pole_mu @ ker_pole_mu.conj().T, transfer_matrix_j_mo_transfer_matrix_j)
    if not cupy.allclose(transfer_matrix_i_j, 0):
        transfer_matrix_i_j = transfer_matrix_i_j / cupy.linalg.norm(transfer_matrix_i_j)
        transfer_matrix[:, i] = cupy.real(transfer_matrix_i_j[:, 0])
        transfer_matrix[:, j] = cupy.imag(transfer_matrix_i_j[:, 0])
    else:
        transfer_matrix[:, i] = cupy.real(ker_pole_mu[:, 0])
        transfer_matrix[:, j] = cupy.imag(ker_pole_mu[:, 0])