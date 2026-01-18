from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class NumSysLin(_NumSys):

    def internal_x0_cb(self, init_concs, params):
        return (99 * init_concs + self.eqsys.dissolved(init_concs)) / 100

    def f(self, yvec, params):
        from pyneqsys.symbolic import linear_exprs
        init_concs, eq_params = self._inits_and_eq_params(params)
        A, ks = self._get_A_ks(eq_params)
        f_equil = [q / k - 1 if k != 0 else q for q, k in zip(prodpow(yvec, A), ks)]
        B, comp_nrs = self.eqsys.composition_balance_vectors()
        f_preserv = linear_exprs(B, yvec, mat_dot_vec(B, init_concs), rref=self.rref_preserv)
        return f_equil + f_preserv