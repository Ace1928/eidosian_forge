import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_ALL(_Test_Utility):
    """
    To test all vector types at once
    """

    def test_memory_load(self):
        data = self._data()
        load_data = self.load(data)
        assert load_data == data
        loada_data = self.loada(data)
        assert loada_data == data
        loads_data = self.loads(data)
        assert loads_data == data
        loadl = self.loadl(data)
        loadl_half = list(loadl)[:self.nlanes // 2]
        data_half = data[:self.nlanes // 2]
        assert loadl_half == data_half
        assert loadl != data

    def test_memory_store(self):
        data = self._data()
        vdata = self.load(data)
        store = [0] * self.nlanes
        self.store(store, vdata)
        assert store == data
        store_a = [0] * self.nlanes
        self.storea(store_a, vdata)
        assert store_a == data
        store_s = [0] * self.nlanes
        self.stores(store_s, vdata)
        assert store_s == data
        store_l = [0] * self.nlanes
        self.storel(store_l, vdata)
        assert store_l[:self.nlanes // 2] == data[:self.nlanes // 2]
        assert store_l != vdata
        store_h = [0] * self.nlanes
        self.storeh(store_h, vdata)
        assert store_h[:self.nlanes // 2] == data[self.nlanes // 2:]
        assert store_h != vdata

    @pytest.mark.parametrize('intrin, elsizes, scale, fill', [('self.load_tillz, self.load_till', (32, 64), 1, [65535]), ('self.load2_tillz, self.load2_till', (32, 64), 2, [65535, 32767])])
    def test_memory_partial_load(self, intrin, elsizes, scale, fill):
        if self._scalar_size() not in elsizes:
            return
        npyv_load_tillz, npyv_load_till = eval(intrin)
        data = self._data()
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes ** 2, self.nlanes ** 4]
        for n in lanes:
            load_till = npyv_load_till(data, n, *fill)
            load_tillz = npyv_load_tillz(data, n)
            n *= scale
            data_till = data[:n] + fill * ((self.nlanes - n) // scale)
            assert load_till == data_till
            data_tillz = data[:n] + [0] * (self.nlanes - n)
            assert load_tillz == data_tillz

    @pytest.mark.parametrize('intrin, elsizes, scale', [('self.store_till', (32, 64), 1), ('self.store2_till', (32, 64), 2)])
    def test_memory_partial_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_store_till = eval(intrin)
        data = self._data()
        data_rev = self._data(reverse=True)
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes ** 2, self.nlanes ** 4]
        for n in lanes:
            data_till = data_rev.copy()
            data_till[:n * scale] = data[:n * scale]
            store_till = self._data(reverse=True)
            npyv_store_till(store_till, n, vdata)
            assert store_till == data_till

    @pytest.mark.parametrize('intrin, elsizes, scale', [('self.loadn', (32, 64), 1), ('self.loadn2', (32, 64), 2)])
    def test_memory_noncont_load(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_loadn = eval(intrin)
        for stride in range(-64, 64):
            if stride < 0:
                data = self._data(stride, -stride * self.nlanes)
                data_stride = list(itertools.chain(*zip(*[data[-i::stride] for i in range(scale, 0, -1)])))
            elif stride == 0:
                data = self._data()
                data_stride = data[0:scale] * (self.nlanes // scale)
            else:
                data = self._data(count=stride * self.nlanes)
                data_stride = list(itertools.chain(*zip(*[data[i::stride] for i in range(scale)])))
            data_stride = self.load(data_stride)
            loadn = npyv_loadn(data, stride)
            assert loadn == data_stride

    @pytest.mark.parametrize('intrin, elsizes, scale, fill', [('self.loadn_tillz, self.loadn_till', (32, 64), 1, [65535]), ('self.loadn2_tillz, self.loadn2_till', (32, 64), 2, [65535, 32767])])
    def test_memory_noncont_partial_load(self, intrin, elsizes, scale, fill):
        if self._scalar_size() not in elsizes:
            return
        npyv_loadn_tillz, npyv_loadn_till = eval(intrin)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes ** 2, self.nlanes ** 4]
        for stride in range(-64, 64):
            if stride < 0:
                data = self._data(stride, -stride * self.nlanes)
                data_stride = list(itertools.chain(*zip(*[data[-i::stride] for i in range(scale, 0, -1)])))
            elif stride == 0:
                data = self._data()
                data_stride = data[0:scale] * (self.nlanes // scale)
            else:
                data = self._data(count=stride * self.nlanes)
                data_stride = list(itertools.chain(*zip(*[data[i::stride] for i in range(scale)])))
            data_stride = list(self.load(data_stride))
            for n in lanes:
                nscale = n * scale
                llanes = self.nlanes - nscale
                data_stride_till = data_stride[:nscale] + fill * (llanes // scale)
                loadn_till = npyv_loadn_till(data, stride, n, *fill)
                assert loadn_till == data_stride_till
                data_stride_tillz = data_stride[:nscale] + [0] * llanes
                loadn_tillz = npyv_loadn_tillz(data, stride, n)
                assert loadn_tillz == data_stride_tillz

    @pytest.mark.parametrize('intrin, elsizes, scale', [('self.storen', (32, 64), 1), ('self.storen2', (32, 64), 2)])
    def test_memory_noncont_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_storen = eval(intrin)
        data = self._data()
        vdata = self.load(data)
        hlanes = self.nlanes // scale
        for stride in range(1, 64):
            data_storen = [255] * stride * self.nlanes
            for s in range(0, hlanes * stride, stride):
                i = s // stride * scale
                data_storen[s:s + scale] = data[i:i + scale]
            storen = [255] * stride * self.nlanes
            storen += [127] * 64
            npyv_storen(storen, stride, vdata)
            assert storen[:-64] == data_storen
            assert storen[-64:] == [127] * 64
        for stride in range(-64, 0):
            data_storen = [255] * -stride * self.nlanes
            for s in range(0, hlanes * stride, stride):
                i = s // stride * scale
                data_storen[s - scale:s or None] = data[i:i + scale]
            storen = [127] * 64
            storen += [255] * -stride * self.nlanes
            npyv_storen(storen, stride, vdata)
            assert storen[64:] == data_storen
            assert storen[:64] == [127] * 64
        data_storen = [127] * self.nlanes
        storen = data_storen.copy()
        data_storen[0:scale] = data[-scale:]
        npyv_storen(storen, 0, vdata)
        assert storen == data_storen

    @pytest.mark.parametrize('intrin, elsizes, scale', [('self.storen_till', (32, 64), 1), ('self.storen2_till', (32, 64), 2)])
    def test_memory_noncont_partial_store(self, intrin, elsizes, scale):
        if self._scalar_size() not in elsizes:
            return
        npyv_storen_till = eval(intrin)
        data = self._data()
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes ** 2, self.nlanes ** 4]
        hlanes = self.nlanes // scale
        for stride in range(1, 64):
            for n in lanes:
                data_till = [255] * stride * self.nlanes
                tdata = data[:n * scale] + [255] * (self.nlanes - n * scale)
                for s in range(0, hlanes * stride, stride)[:n]:
                    i = s // stride * scale
                    data_till[s:s + scale] = tdata[i:i + scale]
                storen_till = [255] * stride * self.nlanes
                storen_till += [127] * 64
                npyv_storen_till(storen_till, stride, n, vdata)
                assert storen_till[:-64] == data_till
                assert storen_till[-64:] == [127] * 64
        for stride in range(-64, 0):
            for n in lanes:
                data_till = [255] * -stride * self.nlanes
                tdata = data[:n * scale] + [255] * (self.nlanes - n * scale)
                for s in range(0, hlanes * stride, stride)[:n]:
                    i = s // stride * scale
                    data_till[s - scale:s or None] = tdata[i:i + scale]
                storen_till = [127] * 64
                storen_till += [255] * -stride * self.nlanes
                npyv_storen_till(storen_till, stride, n, vdata)
                assert storen_till[64:] == data_till
                assert storen_till[:64] == [127] * 64
        for n in lanes:
            data_till = [127] * self.nlanes
            storen_till = data_till.copy()
            data_till[0:scale] = data[:n * scale][-scale:]
            npyv_storen_till(storen_till, 0, n, vdata)
            assert storen_till == data_till

    @pytest.mark.parametrize('intrin, table_size, elsize', [('self.lut32', 32, 32), ('self.lut16', 16, 64)])
    def test_lut(self, intrin, table_size, elsize):
        """
        Test lookup table intrinsics:
            npyv_lut32_##sfx
            npyv_lut16_##sfx
        """
        if elsize != self._scalar_size():
            return
        intrin = eval(intrin)
        idx_itrin = getattr(self.npyv, f'setall_u{elsize}')
        table = range(0, table_size)
        for i in table:
            broadi = self.setall(i)
            idx = idx_itrin(i)
            lut = intrin(table, idx)
            assert lut == broadi

    def test_misc(self):
        broadcast_zero = self.zero()
        assert broadcast_zero == [0] * self.nlanes
        for i in range(1, 10):
            broadcasti = self.setall(i)
            assert broadcasti == [i] * self.nlanes
        data_a, data_b = (self._data(), self._data(reverse=True))
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        vset = self.set(*data_a)
        assert vset == data_a
        vsetf = self.setf(10, *data_a)
        assert vsetf == data_a
        sfxes = ['u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64']
        if self.npyv.simd_f64:
            sfxes.append('f64')
        if self.npyv.simd_f32:
            sfxes.append('f32')
        for sfx in sfxes:
            vec_name = getattr(self, 'reinterpret_' + sfx)(vdata_a).__name__
            assert vec_name == 'npyv_' + sfx
        select_a = self.select(self.cmpeq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_a == data_a
        select_b = self.select(self.cmpneq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_b == data_b
        assert self.extract0(vdata_b) == vdata_b[0]
        self.npyv.cleanup()

    def test_reorder(self):
        data_a, data_b = (self._data(), self._data(reverse=True))
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_a_lo = data_a[:self.nlanes // 2]
        data_b_lo = data_b[:self.nlanes // 2]
        data_a_hi = data_a[self.nlanes // 2:]
        data_b_hi = data_b[self.nlanes // 2:]
        combinel = self.combinel(vdata_a, vdata_b)
        assert combinel == data_a_lo + data_b_lo
        combineh = self.combineh(vdata_a, vdata_b)
        assert combineh == data_a_hi + data_b_hi
        combine = self.combine(vdata_a, vdata_b)
        assert combine == (data_a_lo + data_b_lo, data_a_hi + data_b_hi)
        data_zipl = self.load([v for p in zip(data_a_lo, data_b_lo) for v in p])
        data_ziph = self.load([v for p in zip(data_a_hi, data_b_hi) for v in p])
        vzip = self.zip(vdata_a, vdata_b)
        assert vzip == (data_zipl, data_ziph)
        vzip = [0] * self.nlanes * 2
        self._x2('store')(vzip, (vdata_a, vdata_b))
        assert vzip == list(data_zipl) + list(data_ziph)
        unzip = self.unzip(data_zipl, data_ziph)
        assert unzip == (data_a, data_b)
        unzip = self._x2('load')(list(data_zipl) + list(data_ziph))
        assert unzip == (data_a, data_b)

    def test_reorder_rev64(self):
        ssize = self._scalar_size()
        if ssize == 64:
            return
        data_rev64 = [y for x in range(0, self.nlanes, 64 // ssize) for y in reversed(range(x, x + 64 // ssize))]
        rev64 = self.rev64(self.load(range(self.nlanes)))
        assert rev64 == data_rev64

    def test_reorder_permi128(self):
        """
        Test permuting elements for each 128-bit lane.
        npyv_permi128_##sfx
        """
        ssize = self._scalar_size()
        if ssize < 32:
            return
        data = self.load(self._data())
        permn = 128 // ssize
        permd = permn - 1
        nlane128 = self.nlanes // permn
        shfl = [0, 1] if ssize == 64 else [0, 2, 4, 6]
        for i in range(permn):
            indices = [i >> shf & permd for shf in shfl]
            vperm = self.permi128(data, *indices)
            data_vperm = [data[j + (e & -permn)] for e, j in enumerate(indices * nlane128)]
            assert vperm == data_vperm

    @pytest.mark.parametrize('func, intrin', [(operator.lt, 'cmplt'), (operator.le, 'cmple'), (operator.gt, 'cmpgt'), (operator.ge, 'cmpge'), (operator.eq, 'cmpeq')])
    def test_operators_comparison(self, func, intrin):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        intrin = getattr(self, intrin)
        mask_true = self._true_mask()

        def to_bool(vector):
            return [lane == mask_true for lane in vector]
        data_cmp = [func(a, b) for a, b in zip(data_a, data_b)]
        cmp = to_bool(intrin(vdata_a, vdata_b))
        assert cmp == data_cmp

    def test_operators_logical(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        if self._is_fp():
            data_cast_a = self._to_unsigned(vdata_a)
            data_cast_b = self._to_unsigned(vdata_b)
            cast, cast_data = (self._to_unsigned, self._to_unsigned)
        else:
            data_cast_a, data_cast_b = (data_a, data_b)
            cast, cast_data = (lambda a: a, self.load)
        data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
        vxor = cast(self.xor(vdata_a, vdata_b))
        assert vxor == data_xor
        data_or = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
        vor = cast(getattr(self, 'or')(vdata_a, vdata_b))
        assert vor == data_or
        data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
        vand = cast(getattr(self, 'and')(vdata_a, vdata_b))
        assert vand == data_and
        data_not = cast_data([~a for a in data_cast_a])
        vnot = cast(getattr(self, 'not')(vdata_a))
        assert vnot == data_not
        if self.sfx not in 'u8':
            return
        data_andc = [a & ~b for a, b in zip(data_cast_a, data_cast_b)]
        vandc = cast(getattr(self, 'andc')(vdata_a, vdata_b))
        assert vandc == data_andc

    @pytest.mark.parametrize('intrin', ['any', 'all'])
    @pytest.mark.parametrize('data', ([1, 2, 3, 4], [-1, -2, -3, -4], [0, 1, 2, 3, 4], [127, 32767, 2147483647, 9223372036854775807], [0, -1, -2, -3, 4], [0], [1], [-1]))
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

    def test_conversion_boolean(self):
        bsfx = 'b' + self.sfx[1:]
        to_boolean = getattr(self.npyv, 'cvt_%s_%s' % (bsfx, self.sfx))
        from_boolean = getattr(self.npyv, 'cvt_%s_%s' % (self.sfx, bsfx))
        false_vb = to_boolean(self.setall(0))
        true_vb = self.cmpeq(self.setall(0), self.setall(0))
        assert false_vb != true_vb
        false_vsfx = from_boolean(false_vb)
        true_vsfx = from_boolean(true_vb)
        assert false_vsfx != true_vsfx

    def test_conversion_expand(self):
        """
        Test expand intrinsics:
            npyv_expand_u16_u8
            npyv_expand_u32_u16
        """
        if self.sfx not in ('u8', 'u16'):
            return
        totype = self.sfx[0] + str(int(self.sfx[1:]) * 2)
        expand = getattr(self.npyv, f'expand_{totype}_{self.sfx}')
        data = self._data(self._int_max() - self.nlanes)
        vdata = self.load(data)
        edata = expand(vdata)
        data_lo = data[:self.nlanes // 2]
        data_hi = data[self.nlanes // 2:]
        assert edata == (data_lo, data_hi)

    def test_arithmetic_subadd(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_add = self.load([a + b for a, b in zip(data_a, data_b)])
        add = self.add(vdata_a, vdata_b)
        assert add == data_add
        data_sub = self.load([a - b for a, b in zip(data_a, data_b)])
        sub = self.sub(vdata_a, vdata_b)
        assert sub == data_sub

    def test_arithmetic_mul(self):
        if self.sfx in ('u64', 's64'):
            return
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_mul = self.load([a * b for a, b in zip(data_a, data_b)])
        mul = self.mul(vdata_a, vdata_b)
        assert mul == data_mul

    def test_arithmetic_div(self):
        if not self._is_fp():
            return
        data_a, data_b = (self._data(), self._data(reverse=True))
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_div = self.load([a / b for a, b in zip(data_a, data_b)])
        div = self.div(vdata_a, vdata_b)
        assert div == data_div

    def test_arithmetic_intdiv(self):
        """
        Test integer division intrinsics:
            npyv_divisor_##sfx
            npyv_divc_##sfx
        """
        if self._is_fp():
            return
        int_min = self._int_min()

        def trunc_div(a, d):
            """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
            if d == -1 and a == int_min:
                return a
            sign_a, sign_d = (a < 0, d < 0)
            if a == 0 or sign_a == sign_d:
                return a // d
            return (a + sign_d - sign_a) // d + 1
        data = [1, -int_min]
        data += range(0, 2 ** 8, 2 ** 5)
        data += range(0, 2 ** 8, 2 ** 5 - 1)
        bsize = self._scalar_size()
        if bsize > 8:
            data += range(2 ** 8, 2 ** 16, 2 ** 13)
            data += range(2 ** 8, 2 ** 16, 2 ** 13 - 1)
        if bsize > 16:
            data += range(2 ** 16, 2 ** 32, 2 ** 29)
            data += range(2 ** 16, 2 ** 32, 2 ** 29 - 1)
        if bsize > 32:
            data += range(2 ** 32, 2 ** 64, 2 ** 61)
            data += range(2 ** 32, 2 ** 64, 2 ** 61 - 1)
        data += [-x for x in data]
        for dividend, divisor in itertools.product(data, data):
            divisor = self.setall(divisor)[0]
            if divisor == 0:
                continue
            dividend = self.load(self._data(dividend))
            data_divc = [trunc_div(a, divisor) for a in dividend]
            divisor_parms = self.divisor(divisor)
            divc = self.divc(dividend, divisor_parms)
            assert divc == data_divc

    def test_arithmetic_reduce_sum(self):
        """
        Test reduce sum intrinsics:
            npyv_sum_##sfx
        """
        if self.sfx not in ('u32', 'u64', 'f32', 'f64'):
            return
        data = self._data()
        vdata = self.load(data)
        data_sum = sum(data)
        vsum = self.sum(vdata)
        assert vsum == data_sum

    def test_arithmetic_reduce_sumup(self):
        """
        Test extend reduce sum intrinsics:
            npyv_sumup_##sfx
        """
        if self.sfx not in ('u8', 'u16'):
            return
        rdata = (0, self.nlanes, self._int_min(), self._int_max() - self.nlanes)
        for r in rdata:
            data = self._data(r)
            vdata = self.load(data)
            data_sum = sum(data)
            vsum = self.sumup(vdata)
            assert vsum == data_sum

    def test_mask_conditional(self):
        """
        Conditional addition and subtraction for all supported data types.
        Test intrinsics:
            npyv_ifadd_##SFX, npyv_ifsub_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_b = self.load(self._data(reverse=True))
        true_mask = self.cmpeq(self.zero(), self.zero())
        false_mask = self.cmpneq(self.zero(), self.zero())
        data_sub = self.sub(vdata_b, vdata_a)
        ifsub = self.ifsub(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifsub == data_sub
        ifsub = self.ifsub(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifsub == vdata_b
        data_add = self.add(vdata_b, vdata_a)
        ifadd = self.ifadd(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifadd == data_add
        ifadd = self.ifadd(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifadd == vdata_b
        if not self._is_fp():
            return
        data_div = self.div(vdata_b, vdata_a)
        ifdiv = self.ifdiv(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifdiv == data_div
        ifdivz = self.ifdivz(true_mask, vdata_b, vdata_a)
        assert ifdivz == data_div
        ifdiv = self.ifdiv(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifdiv == vdata_b
        ifdivz = self.ifdivz(false_mask, vdata_a, vdata_b)
        assert ifdivz == self.zero()