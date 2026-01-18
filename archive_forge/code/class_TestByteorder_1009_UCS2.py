import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestByteorder_1009_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value