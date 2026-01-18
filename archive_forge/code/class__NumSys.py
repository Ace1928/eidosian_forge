from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class _NumSys(object):
    small = 0
    pre_processor = None
    post_processor = None
    internal_x0_cb = None

    def __init__(self, eqsys, rref_equil=False, rref_preserv=False, backend=None, precipitates=(), new_eq_params=True):
        self.eqsys = eqsys
        self.rref_equil = rref_equil
        self.rref_preserv = rref_preserv
        self.backend = get_backend(backend)
        self.precipitates = precipitates
        self.new_eq_params = new_eq_params

    def _get_A_ks(self, eq_params):
        non_precip_rids = self.eqsys.non_precip_rids(self.precipitates)
        return self.eqsys.stoichs_constants(self.eqsys.eq_constants(non_precip_rids, eq_params, self.small), self.rref_equil, backend=self.backend, non_precip_rids=non_precip_rids)

    def _inits_and_eq_params(self, params):
        eq_params = params[self.eqsys.ns:]
        if not self.new_eq_params:
            assert not eq_params, 'Adjust number of parameters accordingly'
            eq_params = None
        return (params[:self.eqsys.ns], eq_params)