import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner
def _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full, coef_vals, raveled_offsets):
    """Solve a (sparse) linear system corresponding to biharmonic inpainting.

    This function creates a linear system of the form:

    ``A @ u = b``

    where ``A`` is a sparse matrix, ``b`` is a vector enforcing smoothness and
    boundary constraints and ``u`` is the vector of inpainted values to be
    (uniquely) determined by solving the linear system.

    ``A`` is a sparse matrix of shape (n_mask, n_mask) where ``n_mask``
    corresponds to the number of non-zero values in ``mask`` (i.e. the number
    of pixels to be inpainted). Each row in A will have a number of non-zero
    values equal to the number of non-zero values in the biharmonic kernel,
    ``neigh_coef_full``. In practice, biharmonic kernels with reduced extent
    are used at the image borders. This matrix, ``A`` is the same for all
    image channels (since the same inpainting mask is currently used for all
    channels).

    ``u`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the vector of unknown values for each channel.

    ``b`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the desired output of convolving the solution with the biharmonic kernel.
    At mask locations where there is no overlap with known values, ``b`` will
    have a value of 0. This enforces the biharmonic smoothness constraint in
    the interior of inpainting regions. For regions near the boundary that
    overlap with known values, the entries in ``b`` enforce boundary conditions
    designed to avoid discontinuity with the known values.
    """
    n_channels = out.shape[-1]
    radius = neigh_coef_full.shape[0] // 2
    edge_mask = np.ones(mask.shape, dtype=bool)
    edge_mask[(slice(radius, -radius),) * mask.ndim] = 0
    boundary_mask = edge_mask * mask
    center_mask = ~edge_mask * mask
    boundary_pts = np.where(boundary_mask)
    boundary_i = np.flatnonzero(boundary_mask)
    center_i = np.flatnonzero(center_mask)
    mask_i = np.concatenate((boundary_i, center_i))
    center_pts = np.where(center_mask)
    mask_pts = tuple([np.concatenate((b, c)) for b, c in zip(boundary_pts, center_pts)])
    structure = neigh_coef_full != 0
    tmp = ndi.convolve(mask, structure, output=np.uint8, mode='constant')
    nnz_matrix = tmp[mask].sum()
    n_mask = np.count_nonzero(mask)
    n_struct = np.count_nonzero(structure)
    nnz_rhs_vector_max = n_mask - np.count_nonzero(tmp == n_struct)
    row_idx_known = np.empty(nnz_rhs_vector_max, dtype=np.intp)
    data_known = np.zeros((nnz_rhs_vector_max, n_channels), dtype=out.dtype)
    row_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    col_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    data_unknown = np.empty(nnz_matrix, dtype=out.dtype)
    coef_cache = {}
    mask_flat = mask.reshape(-1)
    out_flat = np.ascontiguousarray(out.reshape((-1, n_channels)))
    idx_known = 0
    idx_unknown = 0
    mask_pt_n = -1
    boundary_pts = np.stack(boundary_pts, axis=1)
    for mask_pt_n, nd_idx in enumerate(boundary_pts):
        b_lo, b_hi = _get_neighborhood(nd_idx, radius, mask.shape)
        coef_shape = tuple(b_hi - b_lo)
        coef_center = tuple(nd_idx - b_lo)
        coef_idx, coefs = coef_cache.get((coef_shape, coef_center), (None, None))
        if coef_idx is None:
            _, coef_idx, coefs = _get_neigh_coef(coef_shape, coef_center, dtype=out.dtype)
            coef_cache[coef_shape, coef_center] = (coef_idx, coefs)
        coef_idx = coef_idx + b_lo[:, np.newaxis]
        index1d = np.ravel_multi_index(coef_idx, mask.shape)
        nvals = 0
        for coef, i in zip(coefs, index1d):
            if mask_flat[i]:
                row_idx_unknown[idx_unknown] = mask_pt_n
                col_idx_unknown[idx_unknown] = i
                data_unknown[idx_unknown] = coef
                idx_unknown += 1
            else:
                data_known[idx_known, :] -= coef * out_flat[i, :]
                nvals += 1
        if nvals:
            row_idx_known[idx_known] = mask_pt_n
            idx_known += 1
    row_start = mask_pt_n + 1
    known_start_idx = idx_known
    unknown_start_idx = idx_unknown
    nnz_rhs = _build_matrix_inner(row_start, known_start_idx, unknown_start_idx, center_i, raveled_offsets, coef_vals, mask_flat, out_flat, row_idx_known, data_known, row_idx_unknown, col_idx_unknown, data_unknown)
    row_idx_known = row_idx_known[:nnz_rhs]
    data_known = data_known[:nnz_rhs, :]
    sp_shape = (n_mask, out.size)
    matrix_unknown = sparse.coo_matrix((data_unknown, (row_idx_unknown, col_idx_unknown)), shape=sp_shape).tocsr()
    matrix_unknown = matrix_unknown[:, mask_i]
    rhs = np.zeros((n_mask, n_channels), dtype=out.dtype)
    rhs[row_idx_known, :] = data_known
    result = spsolve(matrix_unknown, rhs, use_umfpack=False, permc_spec='MMD_ATA')
    if result.ndim == 1:
        result = result[:, np.newaxis]
    out[mask_pts] = result
    return out