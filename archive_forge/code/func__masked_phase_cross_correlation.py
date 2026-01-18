from functools import partial
import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len
from .._shared.utils import _supported_float_type
def _masked_phase_cross_correlation(reference_image, moving_image, reference_mask, moving_mask=None, overlap_ratio=0.3):
    """Masked image translation registration by masked normalized
    cross-correlation.

    Parameters
    ----------
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Image to register. Must be same dimensionality as ``reference_image``,
        but not necessarily the same size.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with numpy.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

    """
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError('Input images have different shapes, moving_mask must be explicitly set.')
        moving_mask = reference_mask.astype(bool)
    for im, mask in [(reference_image, reference_mask), (moving_image, moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError('Image sizes must match their respective mask sizes.')
    xcorr = cross_correlate_masked(moving_image, reference_image, moving_mask, reference_mask, axes=tuple(range(moving_image.ndim)), mode='full', overlap_ratio=overlap_ratio)
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shifts = center - np.array(reference_image.shape) + 1
    size_mismatch = np.array(moving_image.shape) - np.array(reference_image.shape)
    return -shifts + size_mismatch / 2