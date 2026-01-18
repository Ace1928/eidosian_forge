import pytest
import numpy as np
from numpy.testing import (
class TestArrayFromScalar:
    """ gh-15467 """

    def _do_test(self, t1, t2):
        x = t1(2)
        arr = np.array(x, dtype=t2)
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    def test_complex(self, t1, t2):
        return self._do_test(t1, t2)