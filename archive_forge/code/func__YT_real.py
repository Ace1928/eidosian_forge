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
def _YT_real(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.1 page 19 related to real pairs
    """
    i = int(i)
    j = int(j)
    u = Q[:, -2, None]
    v = Q[:, -1, None]
    m = ker_pole[i].T @ (u @ v.T - v @ u.T) @ ker_pole[j]
    um, sm, vm = cupy.linalg.svd(m)
    mu1, mu2 = um.T[:2, :, None]
    nu1, nu2 = vm[:2, :, None]
    transfer_matrix_j_mo_transfer_matrix_j = cupy.vstack((transfer_matrix[:, i, None], transfer_matrix[:, j, None]))
    if not cupy.allclose(sm[0], sm[1]):
        ker_pole_imo_mu1 = ker_pole[i] @ mu1
        ker_pole_i_nu1 = ker_pole[j] @ nu1
        ker_pole_mu_nu = cupy.vstack((ker_pole_imo_mu1, ker_pole_i_nu1))
    else:
        ker_pole_ij = cupy.vstack((cupy.hstack((ker_pole[i], cupy.zeros(ker_pole[i].shape))), cupy.hstack((cupy.zeros(ker_pole[j].shape), ker_pole[j]))))
        mu_nu_matrix = cupy.vstack((cupy.hstack((mu1, mu2)), cupy.hstack((nu1, nu2))))
        ker_pole_mu_nu = ker_pole_ij @ mu_nu_matrix
    transfer_matrix_ij = ker_pole_mu_nu @ ker_pole_mu_nu.T @ transfer_matrix_j_mo_transfer_matrix_j
    if not cupy.allclose(transfer_matrix_ij, 0):
        transfer_matrix_ij = sqrt(2) * transfer_matrix_ij / cupy.linalg.norm(transfer_matrix_ij)
        transfer_matrix[:, i] = transfer_matrix_ij[:transfer_matrix[:, i].shape[0], 0]
        transfer_matrix[:, j] = transfer_matrix_ij[transfer_matrix[:, i].shape[0]:, 0]
    else:
        transfer_matrix[:, i] = ker_pole_mu_nu[:transfer_matrix[:, i].shape[0], 0]
        transfer_matrix[:, j] = ker_pole_mu_nu[transfer_matrix[:, i].shape[0]:, 0]