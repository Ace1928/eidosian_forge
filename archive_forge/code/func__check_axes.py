import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
def _check_axes(self, op, xp):
    x = xp.asarray(random((30, 20, 10)))
    axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    xp_test = array_namespace(x)
    for a in axes:
        op_tr = op(xp_test.permute_dims(x, axes=a))
        tr_op = xp_test.permute_dims(op(x, axes=a), axes=a)
        xp_assert_close(op_tr, tr_op)