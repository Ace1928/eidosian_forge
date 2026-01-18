import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@classmethod
def _calculate_image_empirical_pdf(cls, image):
    """Helper function for calculating empirical probability density
        function of a given image for all channels"""
    if image.ndim > 2:
        image = image.transpose(2, 0, 1)
    channels = np.array(image, copy=False, ndmin=3)
    channels_pdf = []
    for channel in channels:
        channel_values, counts = np.unique(channel, return_counts=True)
        channel_quantiles = np.cumsum(counts).astype(np.float64)
        channel_quantiles /= channel_quantiles[-1]
        channels_pdf.append((channel_values, channel_quantiles))
    return np.asarray(channels_pdf, dtype=object)