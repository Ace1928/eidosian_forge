from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base.boolean_var import _DeprecatedImplicitAssociatedBinaryVariable
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import native_logical_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.util import target_list
def _cnf_to_linear_constraint_list(cnf_expr, indicator_var=None, binary_varlist=None):
    if type(cnf_expr) in native_types or cnf_expr.is_constant():
        if value(cnf_expr) is True:
            return []
        else:
            raise ValueError('Cannot build linear constraint for logical expression with constant value False: %s' % cnf_expr)
    if cnf_expr.is_expression_type():
        return CnfToLinearVisitor(indicator_var, binary_varlist).walk_expression(cnf_expr)
    else:
        return [cnf_expr.get_associated_binary() == 1]