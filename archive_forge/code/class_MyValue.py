import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
class MyValue:
    """
    Minimal indexable object
    """

    def __init__(self, shape):
        self.ndim = 2
        self.shape = shape
        self._v = np.arange(np.prod(shape)).reshape(shape)

    def __getitem__(self, idx):
        return self._v[idx]

    def __array_interface__(self):
        return None

    def __array__(self):
        raise RuntimeError('No array representation')