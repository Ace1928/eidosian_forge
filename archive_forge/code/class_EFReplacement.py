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
class EFReplacement(ExpressionReplacementVisitor):
    """
    This class is a subclass of ExpressionReplacementVisitor.
    It replaces an external function expression in an expression tree with a
    "holder" variable (recorded in a ComponentMap) and sets the initial value
    of the new node on the tree to that of the original node, if it can.

    NOTE: We use an empty substitution map. The EFs to be substituted are
          identified as part of exitNode.
    """

    def __init__(self, trfData, efSet):
        super().__init__(descend_into_named_expressions=True, remove_named_expressions=False)
        self.trfData = trfData
        self.efSet = efSet

    def beforeChild(self, node, child, child_idx):
        descend, result = super().beforeChild(node, child, child_idx)
        if not descend and result.__class__ not in native_types and result.is_variable_type():
            self.trfData.all_variables.add(result)
        return (descend, result)

    def exitNode(self, node, data):
        new_node = super().exitNode(node, data)
        if new_node.__class__ is not ExternalFunctionExpression:
            return new_node
        if self.efSet is not None and new_node._fcn not in self.efSet:
            return new_node
        _output = self.trfData.ef_outputs.add()
        try:
            _output.set_value(value(node))
        except:
            _output.set_value(0)
        self.trfData.truth_models[_output] = new_node
        return _output