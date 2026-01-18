import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
class TExpr(Expr):
    argument_names = ('heat_capacity',)
    parameter_keys = ('energy',)

    def __call__(self, variables, backend=None):
        heat_capacity, = self.all_args(variables, backend=backend)
        energy, = self.all_params(variables, backend=backend)
        return energy / heat_capacity