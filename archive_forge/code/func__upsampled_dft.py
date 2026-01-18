import itertools
import warnings
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy import ndimage as ndi
from ._masked_phase_cross_correlation import _masked_phase_cross_correlation
def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    if not hasattr(upsampled_region_size, '__iter__'):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    elif len(upsampled_region_size) != data.ndim:
        raise ValueError("shape of upsampled region sizes must be equal to input data's number of dimensions.")
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError("number of axis offsets must be equal to input data's number of dimensions.")
    im2pi = 1j * 2 * np.pi
    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))
    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(n_items, upsample_factor)
        kernel = np.exp(-im2pi * kernel)
        kernel = kernel.astype(data.dtype, copy=False)
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data