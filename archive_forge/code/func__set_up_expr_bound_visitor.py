from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _set_up_expr_bound_visitor(self):
    self._expr_bound_visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=False)