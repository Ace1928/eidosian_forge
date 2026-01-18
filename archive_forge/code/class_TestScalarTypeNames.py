import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestScalarTypeNames:
    numeric_types = [np.byte, np.short, np.intc, np.int_, np.longlong, np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong, np.half, np.single, np.double, np.longdouble, np.csingle, np.cdouble, np.clongdouble]

    def test_names_are_unique(self):
        assert len(set(self.numeric_types)) == len(self.numeric_types)
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    @pytest.mark.parametrize('t', numeric_types)
    def test_names_reflect_attributes(self, t):
        """ Test that names correspond to where the type is under ``np.`` """
        assert getattr(np, t.__name__) is t

    @pytest.mark.parametrize('t', numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        """ Test the dtype constructor maps names back to the type """
        assert np.dtype(t.__name__).type is t