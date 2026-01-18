import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestByteorder_1_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value