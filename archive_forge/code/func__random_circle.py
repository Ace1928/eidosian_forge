import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def _random_circle(shape):
    np.random.seed(98312871)
    image = np.random.rand(*shape)
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.sqrt((c0 - shape[0] // 2) ** 2 + (c1 - shape[1] // 2) ** 2)
    radius = min(shape) // 2
    image[r > radius] = 0.0
    return image