from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class _NumSysLinNegPenalty(NumSysLin):

    def f(self, yvec, params):
        import sympy as sp
        f_penalty = [sp.Piecewise((yi ** 2, yi < 0), (0, True)) for yi in yvec]
        return super(_NumSysLinNegPenalty, self).f(yvec, params) + f_penalty