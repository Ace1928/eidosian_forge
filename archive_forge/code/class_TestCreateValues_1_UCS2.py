import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestCreateValues_1_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value