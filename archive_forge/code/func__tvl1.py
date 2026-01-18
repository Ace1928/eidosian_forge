from functools import partial
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian as gaussian_filter
from .._shared.utils import _supported_float_type
from ..transform import warp
from ._optical_flow_utils import _coarse_to_fine, _get_warp_points
def _tvl1(reference_image, moving_image, flow0, attachment, tightness, num_warp, num_iter, tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first grayscale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second grayscale image of the sequence.
    flow0 : ndarray, shape (image0.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    attachment : float
        Attachment parameter. The smaller this parameter is,
        the smoother is the solutions.
    tightness : float
        Tightness parameter. It should have a small value in order to
        maintain attachment and regularization parts in
        correspondence.
    num_warp : int
        Number of times moving_image is warped.
    num_iter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    flow : ndarray, shape (image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    """
    dtype = reference_image.dtype
    grid = np.meshgrid(*[np.arange(n, dtype=dtype) for n in reference_image.shape], indexing='ij', sparse=True)
    dt = 0.5 / reference_image.ndim
    reg_num_iter = 2
    f0 = attachment * tightness
    f1 = dt / tightness
    tol *= reference_image.size
    flow_current = flow_previous = flow0
    g = np.zeros((reference_image.ndim,) + reference_image.shape, dtype=dtype)
    proj = np.zeros((reference_image.ndim, reference_image.ndim) + reference_image.shape, dtype=dtype)
    s_g = [slice(None)] * g.ndim
    s_p = [slice(None)] * proj.ndim
    s_d = [slice(None)] * (proj.ndim - 2)
    for _ in range(num_warp):
        if prefilter:
            flow_current = ndi.median_filter(flow_current, [1] + reference_image.ndim * [3])
        image1_warp = warp(moving_image, _get_warp_points(grid, flow_current), mode='edge')
        grad = np.array(np.gradient(image1_warp))
        NI = (grad * grad).sum(0)
        NI[NI == 0] = 1
        rho_0 = image1_warp - reference_image - (grad * flow_current).sum(0)
        for _ in range(num_iter):
            rho = rho_0 + (grad * flow_current).sum(0)
            idx = abs(rho) <= f0 * NI
            flow_auxiliary = flow_current
            flow_auxiliary[:, idx] -= rho[idx] * grad[:, idx] / NI[idx]
            idx = ~idx
            srho = f0 * np.sign(rho[idx])
            flow_auxiliary[:, idx] -= srho * grad[:, idx]
            flow_current = flow_auxiliary.copy()
            for idx in range(reference_image.ndim):
                s_p[0] = idx
                for _ in range(reg_num_iter):
                    for ax in range(reference_image.ndim):
                        s_g[0] = ax
                        s_g[ax + 1] = slice(0, -1)
                        g[tuple(s_g)] = np.diff(flow_current[idx], axis=ax)
                        s_g[ax + 1] = slice(None)
                    norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
                    norm *= f1
                    norm += 1.0
                    proj[idx] -= dt * g
                    proj[idx] /= norm
                    d = -proj[idx].sum(0)
                    for ax in range(reference_image.ndim):
                        s_p[1] = ax
                        s_p[ax + 2] = slice(0, -1)
                        s_d[ax] = slice(1, None)
                        d[tuple(s_d)] += proj[tuple(s_p)]
                        s_p[ax + 2] = slice(None)
                        s_d[ax] = slice(None)
                    flow_current[idx] = flow_auxiliary[idx] + d
        flow_previous -= flow_current
        if (flow_previous * flow_previous).sum() < tol:
            break
        flow_previous = flow_current
    return flow_current