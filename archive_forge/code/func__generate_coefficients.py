import unittest
import numpy as np
from pygsp import graphs, filters
def _generate_coefficients(self, N, Nf, vertex_delta=83):
    S = np.zeros((N * Nf, Nf))
    S[vertex_delta] = 1
    for i in range(Nf):
        S[vertex_delta + i * self._G.N, i] = 1
    return S