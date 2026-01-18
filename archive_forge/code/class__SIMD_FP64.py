import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_FP64(_Test_Utility):
    """
    To only test double precision
    """

    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        vdata_b = self.mul(vdata_a, self.setall(-1.5))
        data_round = [round(x) for x in list(vdata_a) + list(vdata_b)]
        vround = self.round_s32(vdata_a, vdata_b)
        assert vround == data_round