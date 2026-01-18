import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination
def _remove_ef_from_expr(self, component):
    """
        This method takes a component and looks at its expression.
        If the expression contains an external function (EF), a new expression
        with the EF replaced with a "holder" variable is added to the component
        and the basis expression for the new "holder" variable is updated.
        """
    expr = component.expr
    next_ef_id = len(self.data.ef_outputs)
    new_expr = self.replaceEF(expr)
    if new_expr is not expr:
        component.set_value(new_expr)
        new_output_vars = list((self.data.ef_outputs[i + 1] for i in range(next_ef_id, len(self.data.ef_outputs))))
        for v in new_output_vars:
            self.data.basis_expressions[v] = self.basis_expression_rule(component, self.data.truth_models[v])