import unittest
import numpy as np
from pygsp import graphs, filters
def _test_methods(self, f, tight):
    self.assertIs(f.G, self._G)
    f.evaluate(self._G.e)
    A, B = f.estimate_frame_bounds(use_eigenvalues=True)
    if tight:
        np.testing.assert_allclose(A, B)
    else:
        assert B - A > 0.01
    s2 = f.filter(self._signal, method='exact')
    s3 = f.filter(self._signal, method='chebyshev', order=100)
    s4 = f.filter(s2, method='exact')
    s5 = f.filter(s3, method='chebyshev', order=100)
    if f.Nf < 100:
        np.testing.assert_allclose(s2, s3, rtol=0.1, atol=0.01)
        np.testing.assert_allclose(s4, s5, rtol=0.1, atol=0.01)
    if tight:
        np.testing.assert_allclose(s4, A * self._signal)
        assert np.linalg.norm(s5 - A * self._signal) < 0.1
    self.assertRaises(ValueError, f.filter, s2, method='lanczos')
    if f.Nf < 10:
        F = f.compute_frame(method='exact')
        F = F.reshape(self._G.N, -1)
        s = F.T.dot(self._signal).reshape(self._G.N, -1).squeeze()
        np.testing.assert_allclose(s, s2)
        F = f.compute_frame(method='chebyshev', order=100)
        F = F.reshape(self._G.N, -1)
        s = F.T.dot(self._signal).reshape(self._G.N, -1).squeeze()
        np.testing.assert_allclose(s, s3)