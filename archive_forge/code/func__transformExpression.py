import logging
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
def _transformExpression(self, expr, instance):
    if expr.polynomial_degree() > 2:
        raise ValueError('Cannot transform polynomial terms with degree > 2')
    if expr.polynomial_degree() < 2:
        return expr
    expr = self._replace_bilinear(expr, instance)
    return expr