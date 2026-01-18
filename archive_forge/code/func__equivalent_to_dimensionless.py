import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _equivalent_to_dimensionless(self, a, TOL=1e-12):
    if a is self._pint_dimensionless or a == self._pint_dimensionless:
        return True
    base_a = self._pint_registry.get_base_units(a)
    if not base_a[1].dimensionless:
        return False
    return self._rel_diff(base_a[0], 1.0) <= TOL