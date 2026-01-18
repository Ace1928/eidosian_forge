import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_FP32(_Test_Utility):
    """
    To only test single precision
    """

    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        features = self._cpu_features()
        if not self.npyv.simd_f64 and re.match('.*(NEON|ASIMD)', features):
            _round = lambda v: int(v + (0.5 if v >= 0 else -0.5))
        else:
            _round = round
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        data_round = [_round(x) for x in vdata_a]
        vround = self.round_s32(vdata_a)
        assert vround == data_round