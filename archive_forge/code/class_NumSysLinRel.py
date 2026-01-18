from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class NumSysLinRel(NumSysLin):

    def max_concs(self, params, min_=min, dtype=np.float64):
        init_concs = params[:self.eqsys.ns]
        return self.eqsys.upper_conc_bounds(init_concs, min_=min_, dtype=dtype)

    def pre_processor(self, x, params):
        return (x / self.max_concs(params), params)

    def post_processor(self, x, params):
        return (x * self.max_concs(params), params)

    def f(self, yvec, params):
        import sympy as sp
        return NumSysLin.f(self, [m * yi for m, yi in zip(self.max_concs(params, min_=lambda x: sp.Min(*x), dtype=object), yvec)], params)