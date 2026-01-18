from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
class TestDSTIIInt(_TestDSTBase):

    def setup_method(self):
        self.rdt = int
        self.dec = 6
        self.type = 2