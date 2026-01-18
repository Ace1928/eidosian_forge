from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class NumSysSquare(NumSysLin):
    small = 1e-35

    def pre_processor(self, x, params):
        return (np.sqrt(np.abs(x)), params)

    def post_processor(self, x, params):
        return (x ** 2, params)

    def internal_x0_cb(self, init_concs, params):
        return np.sqrt(np.abs(init_concs))

    def f(self, yvec, params):
        ysq = [yi * yi for yi in yvec]
        return NumSysLin.f(self, ysq, params)