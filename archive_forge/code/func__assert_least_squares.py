import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def _assert_least_squares(tf, src, dst):
    baseline = np.sum((tf(src) - dst) ** 2)
    for i in range(tf.params.size):
        for update in [0.001, -0.001]:
            params = np.copy(tf.params)
            params.flat[i] += update
            new_tf = tf.__class__(matrix=params)
            new_ssq = np.sum((new_tf(src) - dst) ** 2)
            assert new_ssq > baseline