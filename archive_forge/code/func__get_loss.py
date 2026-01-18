from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def _get_loss(truth, guess):
    return numpy.sum(numpy.sum(0.5 * numpy.square(truth - guess), 1))