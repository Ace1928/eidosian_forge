from functools import partial
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian as gaussian_filter
from .._shared.utils import _supported_float_type
from ..transform import warp
from ._optical_flow_utils import _coarse_to_fine, _get_warp_points
def _ilk(reference_image, moving_image, flow0, radius, num_warp, gaussian, prefilter):
    """Iterative Lucas-Kanade (iLK) solver for optical flow estimation.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first grayscale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second grayscale image of the sequence.
    flow0 : ndarray, shape (reference_image.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    radius : int
        Radius of the window considered around each pixel.
    num_warp : int
        Number of times moving_image is warped.
    gaussian : bool
        if True, a gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp. This helps to remove potential outliers.

    Returns
    -------
    flow : ndarray, shape (reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    """
    dtype = reference_image.dtype
    ndim = reference_image.ndim
    size = 2 * radius + 1
    if gaussian:
        sigma = ndim * (size / 4,)
        filter_func = partial(gaussian_filter, sigma=sigma, mode='mirror')
    else:
        filter_func = partial(ndi.uniform_filter, size=ndim * (size,), mode='mirror')
    flow = flow0
    A = np.zeros(reference_image.shape + (ndim, ndim), dtype=dtype)
    b = np.zeros(reference_image.shape + (ndim, 1), dtype=dtype)
    grid = np.meshgrid(*[np.arange(n, dtype=dtype) for n in reference_image.shape], indexing='ij', sparse=True)
    for _ in range(num_warp):
        if prefilter:
            flow = ndi.median_filter(flow, (1,) + ndim * (3,))
        moving_image_warp = warp(moving_image, _get_warp_points(grid, flow), mode='edge')
        grad = np.stack(np.gradient(moving_image_warp), axis=0)
        error_image = (grad * flow).sum(axis=0) + reference_image - moving_image_warp
        for i, j in combinations_with_replacement(range(ndim), 2):
            A[..., i, j] = A[..., j, i] = filter_func(grad[i] * grad[j])
        for i in range(ndim):
            b[..., i, 0] = filter_func(grad[i] * error_image)
        idx = abs(np.linalg.det(A)) < 1e-14
        A[idx] = np.eye(ndim, dtype=dtype)
        b[idx] = 0
        flow = np.moveaxis(np.linalg.solve(A, b)[..., 0], ndim, 0)
    return flow