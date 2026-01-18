import numpy as np
from numpy.testing import assert_array_equal
from skimage import color, data, morphology
from skimage.morphology import binary, isotropic
from skimage.util import img_as_bool
def _disk_with_spacing(radius, dtype=np.uint8, *, strict_radius=True, spacing=None):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    if spacing is not None:
        X *= spacing[1]
        Y *= spacing[0]
    if not strict_radius:
        radius += 0.5
    return np.array(X ** 2 + Y ** 2 <= radius ** 2, dtype=dtype)