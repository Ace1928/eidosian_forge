import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def _make_random(self, npts=20):
    np.random.seed(1234)
    xi = np.sort(np.random.random(npts))
    yi = np.random.random(npts)
    return (pchip(xi, yi), xi, yi)