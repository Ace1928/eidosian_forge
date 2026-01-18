from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
def _body_function(self, r1, r2, x):
    """A function that defines the body expression"""
    alpha = self.alpha
    return sum((xi ** 2 for xi in x)) ** 0.5 - (r1 / alpha) ** alpha * (r2 / (1 - alpha)) ** (1 - alpha)