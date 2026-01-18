import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestCreateValues_1009_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value