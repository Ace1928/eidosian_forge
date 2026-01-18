import numpy as np
import numba
from numba.tests.support import TestCase
class Issue455(object):
    """
    Test code from issue 455.
    """

    def __init__(self):
        self.f = []

    def create_f(self):
        code = '\n        def f(x):\n            n = x.shape[0]\n            for i in range(n):\n                x[i] = 1.\n        '
        d = {}
        exec(code.strip(), d)
        self.f.append(numba.jit('void(f8[:])', nopython=True)(d['f']))

    def call_f(self):
        a = np.zeros(10)
        for f in self.f:
            f(a)
        return a