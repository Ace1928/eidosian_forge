import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestCommonType:

    def test_basic(self):
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        acd = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.csingle)
        assert_(common_type(acd) == np.cdouble)