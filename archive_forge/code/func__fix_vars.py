from pyomo.core.expr import ExpressionBase, as_numeric
from pyomo.core import Constraint, Objective, TransformationFactory
from pyomo.core.base.var import Var, _VarData
from pyomo.core.util import sequence
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
def _fix_vars(self, expr, model):
    """Walk through the S-expression, fixing variables."""
    if expr._args is None:
        return expr
    _args = []
    for i in range(len(expr._args)):
        if isinstance(expr._args[i], ExpressionBase):
            _args.append(self._fix_vars(expr._args[i], model))
        elif (isinstance(expr._args[i], Var) or isinstance(expr._args[i], _VarData)) and expr._args[i].fixed:
            if expr._args[i].value != 0.0:
                _args.append(as_numeric(expr._args[i].value))
        else:
            _args.append(expr._args[i])
    expr._args = _args
    return expr