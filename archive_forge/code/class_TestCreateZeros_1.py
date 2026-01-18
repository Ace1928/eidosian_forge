import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestCreateZeros_1(CreateZeros):
    """Check the creation of zero-valued arrays (size 1)"""
    ulen = 1