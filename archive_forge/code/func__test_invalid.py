import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def _test_invalid(bad_format):
    assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))