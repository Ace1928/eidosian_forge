import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _equivalent_pint_units(self, a, b, TOL=1e-12):
    if a is b or a == b:
        return True
    base_a = self._pint_registry.get_base_units(a)
    base_b = self._pint_registry.get_base_units(b)
    if base_a[1] != base_b[1]:
        uc_a = base_a[1].dimensionality
        uc_b = base_b[1].dimensionality
        for key in uc_a.keys() | uc_b.keys():
            if self._rel_diff(uc_a.get(key, 0), uc_b.get(key, 0)) >= TOL:
                return False
    return self._rel_diff(base_a[0], base_b[0]) <= TOL