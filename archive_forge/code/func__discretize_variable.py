from pyomo.core.expr import ProductExpression, PowExpression
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Binary, value
from pyomo.core.base import (
from pyomo.core.base.var import _VarData
import logging
def _discretize_variable(self, b, v, idx):
    _lb, _ub = v.bounds
    if _lb is None or _ub is None:
        raise RuntimeError("Couldn't discretize variable %s: missing finite lower/upper bounds." % v.name)
    _c = Constraint(expr=v == _lb + (_ub - _lb) * (b.dv[idx] + sum((b.z[idx, k] * 2 ** (-k) for k in b.DISCRETIZATION))))
    b.add_component('c_discr_v%s' % idx, _c)