import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_FP(_Test_Utility):
    """
    To test all float vector types at once
    """

    def test_arithmetic_fused(self):
        vdata_a, vdata_b, vdata_c = [self.load(self._data())] * 3
        vdata_cx2 = self.add(vdata_c, vdata_c)
        data_fma = self.load([a * b + c for a, b, c in zip(vdata_a, vdata_b, vdata_c)])
        fma = self.muladd(vdata_a, vdata_b, vdata_c)
        assert fma == data_fma
        fms = self.mulsub(vdata_a, vdata_b, vdata_c)
        data_fms = self.sub(data_fma, vdata_cx2)
        assert fms == data_fms
        nfma = self.nmuladd(vdata_a, vdata_b, vdata_c)
        data_nfma = self.sub(vdata_cx2, data_fma)
        assert nfma == data_nfma
        nfms = self.nmulsub(vdata_a, vdata_b, vdata_c)
        data_nfms = self.mul(data_fma, self.setall(-1))
        assert nfms == data_nfms
        fmas = list(self.muladdsub(vdata_a, vdata_b, vdata_c))
        assert fmas[0::2] == list(data_fms)[0::2]
        assert fmas[1::2] == list(data_fma)[1::2]

    def test_abs(self):
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        data = self._data()
        vdata = self.load(self._data())
        abs_cases = ((-0, 0), (ninf, pinf), (pinf, pinf), (nan, nan))
        for case, desired in abs_cases:
            data_abs = [desired] * self.nlanes
            vabs = self.abs(self.setall(case))
            assert vabs == pytest.approx(data_abs, nan_ok=True)
        vabs = self.abs(self.mul(vdata, self.setall(-1)))
        assert vabs == data

    def test_sqrt(self):
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        data = self._data()
        vdata = self.load(self._data())
        sqrt_cases = ((-0.0, -0.0), (0.0, 0.0), (-1.0, nan), (ninf, nan), (pinf, pinf))
        for case, desired in sqrt_cases:
            data_sqrt = [desired] * self.nlanes
            sqrt = self.sqrt(self.setall(case))
            assert sqrt == pytest.approx(data_sqrt, nan_ok=True)
        data_sqrt = self.load([math.sqrt(x) for x in data])
        sqrt = self.sqrt(vdata)
        assert sqrt == data_sqrt

    def test_square(self):
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        data = self._data()
        vdata = self.load(self._data())
        square_cases = ((nan, nan), (pinf, pinf), (ninf, pinf))
        for case, desired in square_cases:
            data_square = [desired] * self.nlanes
            square = self.square(self.setall(case))
            assert square == pytest.approx(data_square, nan_ok=True)
        data_square = [x * x for x in data]
        square = self.square(vdata)
        assert square == data_square

    @pytest.mark.parametrize('intrin, func', [('ceil', math.ceil), ('trunc', math.trunc), ('floor', math.floor), ('rint', round)])
    def test_rounding(self, intrin, func):
        """
        Test intrinsics:
            npyv_rint_##SFX
            npyv_ceil_##SFX
            npyv_trunc_##SFX
            npyv_floor##SFX
        """
        intrin_name = intrin
        intrin = getattr(self, intrin)
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        round_cases = ((nan, nan), (pinf, pinf), (ninf, ninf))
        for case, desired in round_cases:
            data_round = [desired] * self.nlanes
            _round = intrin(self.setall(case))
            assert _round == pytest.approx(data_round, nan_ok=True)
        for x in range(0, 2 ** 20, 256 ** 2):
            for w in (-1.05, -1.1, -1.15, 1.05, 1.1, 1.15):
                data = self.load([(x + a) * w for a in range(self.nlanes)])
                data_round = [func(x) for x in data]
                _round = intrin(data)
                assert _round == data_round
        for i in (1.1529215045988576e+18, 4.6116860183954304e+18, 5.902958103546122e+20, 2.3611832414184488e+21):
            x = self.setall(i)
            y = intrin(x)
            data_round = [func(n) for n in x]
            assert y == data_round
        if intrin_name == 'floor':
            data_szero = (-0.0,)
        else:
            data_szero = (-0.0, -0.25, -0.3, -0.45, -0.5)
        for w in data_szero:
            _round = self._to_unsigned(intrin(self.setall(w)))
            data_round = self._to_unsigned(self.setall(-0.0))
            assert _round == data_round

    @pytest.mark.parametrize('intrin', ['max', 'maxp', 'maxn', 'min', 'minp', 'minn'])
    def test_max_min(self, intrin):
        """
        Test intrinsics:
            npyv_max_##sfx
            npyv_maxp_##sfx
            npyv_maxn_##sfx
            npyv_min_##sfx
            npyv_minp_##sfx
            npyv_minn_##sfx
            npyv_reduce_max_##sfx
            npyv_reduce_maxp_##sfx
            npyv_reduce_maxn_##sfx
            npyv_reduce_min_##sfx
            npyv_reduce_minp_##sfx
            npyv_reduce_minn_##sfx
        """
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        chk_nan = {'xp': 1, 'np': 1, 'nn': 2, 'xn': 2}.get(intrin[-2:], 0)
        func = eval(intrin[:3])
        reduce_intrin = getattr(self, 'reduce_' + intrin)
        intrin = getattr(self, intrin)
        hf_nlanes = self.nlanes // 2
        cases = (([0.0, -0.0], [-0.0, 0.0]), ([10, -10], [10, -10]), ([pinf, 10], [10, ninf]), ([10, pinf], [ninf, 10]), ([10, -10], [10, -10]), ([-10, 10], [-10, 10]))
        for op1, op2 in cases:
            vdata_a = self.load(op1 * hf_nlanes)
            vdata_b = self.load(op2 * hf_nlanes)
            data = func(vdata_a, vdata_b)
            simd = intrin(vdata_a, vdata_b)
            assert simd == data
            data = func(vdata_a)
            simd = reduce_intrin(vdata_a)
            assert simd == data
        if not chk_nan:
            return
        if chk_nan == 1:
            test_nan = lambda a, b: b if math.isnan(a) else a if math.isnan(b) else b
        else:
            test_nan = lambda a, b: nan if math.isnan(a) or math.isnan(b) else b
        cases = ((nan, 10), (10, nan), (nan, pinf), (pinf, nan), (nan, nan))
        for op1, op2 in cases:
            vdata_ab = self.load([op1, op2] * hf_nlanes)
            data = test_nan(op1, op2)
            simd = reduce_intrin(vdata_ab)
            assert simd == pytest.approx(data, nan_ok=True)
            vdata_a = self.setall(op1)
            vdata_b = self.setall(op2)
            data = [data] * self.nlanes
            simd = intrin(vdata_a, vdata_b)
            assert simd == pytest.approx(data, nan_ok=True)

    def test_reciprocal(self):
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        data = self._data()
        vdata = self.load(self._data())
        recip_cases = ((nan, nan), (pinf, 0.0), (ninf, -0.0), (0.0, pinf), (-0.0, ninf))
        for case, desired in recip_cases:
            data_recip = [desired] * self.nlanes
            recip = self.recip(self.setall(case))
            assert recip == pytest.approx(data_recip, nan_ok=True)
        data_recip = self.load([1 / x for x in data])
        recip = self.recip(vdata)
        assert recip == data_recip

    def test_special_cases(self):
        """
        Compare Not NaN. Test intrinsics:
            npyv_notnan_##SFX
        """
        nnan = self.notnan(self.setall(self._nan()))
        assert nnan == [0] * self.nlanes

    @pytest.mark.parametrize('intrin_name', ['rint', 'trunc', 'ceil', 'floor'])
    def test_unary_invalid_fpexception(self, intrin_name):
        intrin = getattr(self, intrin_name)
        for d in [float('nan'), float('inf'), -float('inf')]:
            v = self.setall(d)
            clear_floatstatus()
            intrin(v)
            assert check_floatstatus(invalid=True) == False

    @pytest.mark.parametrize('py_comp,np_comp', [(operator.lt, 'cmplt'), (operator.le, 'cmple'), (operator.gt, 'cmpgt'), (operator.ge, 'cmpge'), (operator.eq, 'cmpeq'), (operator.ne, 'cmpneq')])
    def test_comparison_with_nan(self, py_comp, np_comp):
        pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
        mask_true = self._true_mask()

        def to_bool(vector):
            return [lane == mask_true for lane in vector]
        intrin = getattr(self, np_comp)
        cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan), (ninf, nan), (-0.0, +0.0))
        for case_operand1, case_operand2 in cmp_cases:
            data_a = [case_operand1] * self.nlanes
            data_b = [case_operand2] * self.nlanes
            vdata_a = self.setall(case_operand1)
            vdata_b = self.setall(case_operand2)
            vcmp = to_bool(intrin(vdata_a, vdata_b))
            data_cmp = [py_comp(a, b) for a, b in zip(data_a, data_b)]
            assert vcmp == data_cmp

    @pytest.mark.parametrize('intrin', ['any', 'all'])
    @pytest.mark.parametrize('data', ([float('nan'), 0], [0, float('nan')], [float('nan'), 1], [1, float('nan')], [float('nan'), float('nan')], [0.0, -0.0], [-0.0, 0.0], [1.0, -0.0]))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired