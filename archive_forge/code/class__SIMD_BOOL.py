import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_BOOL(_Test_Utility):
    """
    To test all boolean vector types at once
    """

    def _nlanes(self):
        return getattr(self.npyv, 'nlanes_u' + self.sfx[1:])

    def _data(self, start=None, count=None, reverse=False):
        true_mask = self._true_mask()
        rng = range(self._nlanes())
        if reverse:
            rng = reversed(rng)
        return [true_mask if x % 2 else 0 for x in rng]

    def _load_b(self, data):
        len_str = self.sfx[1:]
        load = getattr(self.npyv, 'load_u' + len_str)
        cvt = getattr(self.npyv, f'cvt_b{len_str}_u{len_str}')
        return cvt(load(data))

    def test_operators_logical(self):
        """
        Logical operations for boolean types.
        Test intrinsics:
            npyv_xor_##SFX, npyv_and_##SFX, npyv_or_##SFX, npyv_not_##SFX,
            npyv_andc_b8, npvy_orc_b8, nvpy_xnor_b8
        """
        data_a = self._data()
        data_b = self._data(reverse=True)
        vdata_a = self._load_b(data_a)
        vdata_b = self._load_b(data_b)
        data_and = [a & b for a, b in zip(data_a, data_b)]
        vand = getattr(self, 'and')(vdata_a, vdata_b)
        assert vand == data_and
        data_or = [a | b for a, b in zip(data_a, data_b)]
        vor = getattr(self, 'or')(vdata_a, vdata_b)
        assert vor == data_or
        data_xor = [a ^ b for a, b in zip(data_a, data_b)]
        vxor = getattr(self, 'xor')(vdata_a, vdata_b)
        assert vxor == data_xor
        vnot = getattr(self, 'not')(vdata_a)
        assert vnot == data_b
        if self.sfx not in 'b8':
            return
        data_andc = [a & ~b & 255 for a, b in zip(data_a, data_b)]
        vandc = getattr(self, 'andc')(vdata_a, vdata_b)
        assert data_andc == vandc
        data_orc = [(a | ~b) & 255 for a, b in zip(data_a, data_b)]
        vorc = getattr(self, 'orc')(vdata_a, vdata_b)
        assert data_orc == vorc
        data_xnor = [~(a ^ b) & 255 for a, b in zip(data_a, data_b)]
        vxnor = getattr(self, 'xnor')(vdata_a, vdata_b)
        assert data_xnor == vxnor

    def test_tobits(self):
        data2bits = lambda data: sum([int(x != 0) << i for i, x in enumerate(data, 0)])
        for data in (self._data(), self._data(reverse=True)):
            vdata = self._load_b(data)
            data_bits = data2bits(data)
            tobits = self.tobits(vdata)
            bin_tobits = bin(tobits)
            assert bin_tobits == bin(data_bits)

    def test_pack(self):
        """
        Pack multiple vectors into one
        Test intrinsics:
            npyv_pack_b8_b16
            npyv_pack_b8_b32
            npyv_pack_b8_b64
        """
        if self.sfx not in ('b16', 'b32', 'b64'):
            return
        data = self._data()
        rdata = self._data(reverse=True)
        vdata = self._load_b(data)
        vrdata = self._load_b(rdata)
        pack_simd = getattr(self.npyv, f'pack_b8_{self.sfx}')
        if self.sfx == 'b16':
            spack = [i & 255 for i in list(rdata) + list(data)]
            vpack = pack_simd(vrdata, vdata)
        elif self.sfx == 'b32':
            spack = [i & 255 for i in 2 * list(rdata) + 2 * list(data)]
            vpack = pack_simd(vrdata, vrdata, vdata, vdata)
        elif self.sfx == 'b64':
            spack = [i & 255 for i in 4 * list(rdata) + 4 * list(data)]
            vpack = pack_simd(vrdata, vrdata, vrdata, vrdata, vdata, vdata, vdata, vdata)
        assert vpack == spack

    @pytest.mark.parametrize('intrin', ['any', 'all'])
    @pytest.mark.parametrize('data', ([-1, 0], [0, -1], [-1], [0]))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self._load_b(data * self._nlanes())
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired