import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _Test_Utility:
    npyv = None
    sfx = None
    target_name = None

    def __getattr__(self, attr):
        """
        To call NPV intrinsics without the attribute 'npyv' and
        auto suffixing intrinsics according to class attribute 'sfx'
        """
        return getattr(self.npyv, attr + '_' + self.sfx)

    def _x2(self, intrin_name):
        return getattr(self.npyv, f'{intrin_name}_{self.sfx}x2')

    def _data(self, start=None, count=None, reverse=False):
        """
        Create list of consecutive numbers according to number of vector's lanes.
        """
        if start is None:
            start = 1
        if count is None:
            count = self.nlanes
        rng = range(start, start + count)
        if reverse:
            rng = reversed(rng)
        if self._is_fp():
            return [x / 1.0 for x in rng]
        return list(rng)

    def _is_unsigned(self):
        return self.sfx[0] == 'u'

    def _is_signed(self):
        return self.sfx[0] == 's'

    def _is_fp(self):
        return self.sfx[0] == 'f'

    def _scalar_size(self):
        return int(self.sfx[1:])

    def _int_clip(self, seq):
        if self._is_fp():
            return seq
        max_int = self._int_max()
        min_int = self._int_min()
        return [min(max(v, min_int), max_int) for v in seq]

    def _int_max(self):
        if self._is_fp():
            return None
        max_u = self._to_unsigned(self.setall(-1))[0]
        if self._is_signed():
            return max_u // 2
        return max_u

    def _int_min(self):
        if self._is_fp():
            return None
        if self._is_unsigned():
            return 0
        return -(self._int_max() + 1)

    def _true_mask(self):
        max_unsig = getattr(self.npyv, 'setall_u' + self.sfx[1:])(-1)
        return max_unsig[0]

    def _to_unsigned(self, vector):
        if isinstance(vector, (list, tuple)):
            return getattr(self.npyv, 'load_u' + self.sfx[1:])(vector)
        else:
            sfx = vector.__name__.replace('npyv_', '')
            if sfx[0] == 'b':
                cvt_intrin = 'cvt_u{0}_b{0}'
            else:
                cvt_intrin = 'reinterpret_u{0}_{1}'
            return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)

    def _pinfinity(self):
        return float('inf')

    def _ninfinity(self):
        return -float('inf')

    def _nan(self):
        return float('nan')

    def _cpu_features(self):
        target = self.target_name
        if target == 'baseline':
            target = __cpu_baseline__
        else:
            target = target.split('__')
        return ' '.join(target)