import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def _verify_range(msg, x, vmin, vmax, dtype):
    assert_equal(x[0], vmin)
    assert_equal(x[-1], vmax)
    assert x.dtype == dtype