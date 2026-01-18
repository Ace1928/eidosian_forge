from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def _as_exact_matrices(self, optimize=None):
    if optimize is None:
        optimize = self._field[True] is not None
    if len(self) % 4:
        raise ValueError('Not right number of values to form 2x2 matrices')
    K, z, ans = self._field[optimize]
    return (z, [matrix(K, 2, 2, ans[n:n + 4]) for n in range(0, len(ans), 4)])