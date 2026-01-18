import math
import numbers
import numpy as np
from .._shared.utils import _supported_float_type
from ..color.adapt_rgb import adapt_rgb, hsv_value
from .exposure import rescale_intensity
from ..util import img_as_uint
def _clahe(image, kernel_size, clip_limit, nbins):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : (M[, ...]) ndarray
        Input image.
    kernel_size : int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit between 0 and 1 (higher values give more
        contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (M[, ...]) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (e.g. 128) speeds up processing and still produces
    an output image of good quality. A clip limit of 0 or larger than or equal
    to 1 results in standard (non-contrast limited) AHE.
    """
    ndim = image.ndim
    dtype = image.dtype
    pad_start_per_dim = [k // 2 for k in kernel_size]
    pad_end_per_dim = [(k - s % k) % k + int(np.ceil(k / 2.0)) for k, s in zip(kernel_size, image.shape)]
    image = np.pad(image, [[p_i, p_f] for p_i, p_f in zip(pad_start_per_dim, pad_end_per_dim)], mode='reflect')
    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY, dtype=np.min_scalar_type(NR_OF_GRAY))
    lut //= bin_size
    image = lut[image]
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array([np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k) for k, n in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((math.prod(ns_hist), -1))
    kernel_elements = math.prod(kernel_size)
    if clip_limit > 0.0:
        clim = int(np.clip(clip_limit * kernel_elements, 1, None))
    else:
        clim = kernel_elements
    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nbins)
    hist = np.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, kernel_elements)
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))
    map_array = np.pad(hist, [[1, 1] for _ in range(ndim)] + [[0, 0]], mode='edge')
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array([np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (math.prod(ns_proc), math.prod(blocks.shape[ndim:])))
    coeffs = np.meshgrid(*tuple([np.arange(k) / k for k in kernel_size[::-1]]), indexing='ij')
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for dim, c in enumerate(coeffs)]
    result = np.zeros(blocks.shape, dtype=np.float32)
    for iedge, edge in enumerate(np.ndindex(*[2] * ndim)):
        edge_maps = map_array[tuple([slice(e, e + n) for e, n in zip(edge, ns_proc)])]
        edge_maps = edge_maps.reshape((math.prod(ns_proc), -1))
        edge_mapped = np.take_along_axis(edge_maps, blocks, axis=-1)
        edge_coeffs = np.prod([[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])], 0)
        result += (edge_mapped * edge_coeffs).astype(result.dtype)
    result = result.astype(dtype)
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order = np.array([np.arange(0, ndim), np.arange(ndim, ndim * 2)]).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    result = result.reshape(image.shape)
    unpad_slices = tuple([slice(p_i, s - p_f) for p_i, p_f, s in zip(pad_start_per_dim, pad_end_per_dim, image.shape)])
    result = result[unpad_slices]
    return result