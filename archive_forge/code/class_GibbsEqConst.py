import math
from ..util.pyutil import deprecated
from ..util._expr import Expr
class GibbsEqConst(MassActionEq):
    argument_names = ('dH_over_R', 'dS_over_R')
    parameter_keys = ('temperature',)

    def eq_const(self, variables, backend=math, **kwargs):
        dH_over_R, dS_over_R = self.all_args(variables, backend=backend)
        T, = self.all_params(variables, backend=backend)
        return backend.exp(dS_over_R - dH_over_R / T)