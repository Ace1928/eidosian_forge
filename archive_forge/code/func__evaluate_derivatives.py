import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _evaluate_derivatives(self, x, der=None):
    n = self.n
    r = self.r
    if der is None:
        der = self.n
    pi = cupy.zeros((n, len(x)))
    w = cupy.zeros((n, len(x)))
    pi[0] = 1
    p = cupy.zeros((len(x), self.r), dtype=self.dtype)
    p += self.c[0, cupy.newaxis, :]
    for k in range(1, n):
        w[k - 1] = x - self.xi[k - 1]
        pi[k] = w[k - 1] * pi[k - 1]
        p += pi[k, :, cupy.newaxis] * self.c[k]
    cn = cupy.zeros((max(der, n + 1), len(x), r), dtype=self.dtype)
    cn[:n + 1, :, :] += self.c[:n + 1, cupy.newaxis, :]
    cn[0] = p
    for k in range(1, n):
        for i in range(1, n - k + 1):
            pi[i] = w[k + i - 1] * pi[i - 1] + pi[i]
            cn[k] = cn[k] + pi[i, :, cupy.newaxis] * cn[k + i]
        cn[k] *= float_factorial(k)
    cn[n, :, :] = 0
    return cn[:der]