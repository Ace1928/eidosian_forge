import math
import copy
import re
import io
import pyomo.environ as pyo
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr import (
from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet
from pyomo.core.expr.template_expr import (
from pyomo.core.expr.numeric_expr import NPV_SumExpression, NPV_DivisionExpression
from pyomo.core.base.block import IndexedBlock
from pyomo.core.base.external import _PythonCallbackFunctionID
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.block import _BlockData
from pyomo.repn.util import ExprType
from pyomo.common import DeveloperError
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.dependencies import numpy as np, numpy_available
class _LatexVisitor(StreamBasedExpressionVisitor):

    def __init__(self):
        super().__init__()
        self.externalFunctionCounter = 0
        self._operator_handles = {ScalarVar: handle_var_node, int: handle_num_node, float: handle_num_node, NegationExpression: handle_negation_node, ProductExpression: handle_product_node, DivisionExpression: handle_division_node, PowExpression: handle_pow_node, AbsExpression: handle_abs_node, UnaryFunctionExpression: handle_unary_node, Expr_ifExpression: handle_exprif_node, EqualityExpression: handle_equality_node, InequalityExpression: handle_inequality_node, RangedExpression: handle_ranged_inequality_node, _GeneralExpressionData: handle_named_expression_node, ScalarExpression: handle_named_expression_node, kernel.expression.expression: handle_named_expression_node, kernel.expression.noclone: handle_named_expression_node, _GeneralObjectiveData: handle_named_expression_node, _GeneralVarData: handle_var_node, ScalarObjective: handle_named_expression_node, kernel.objective.objective: handle_named_expression_node, ExternalFunctionExpression: handle_external_function_node, _PythonCallbackFunctionID: handle_functionID_node, LinearExpression: handle_sumExpression_node, SumExpression: handle_sumExpression_node, MonomialTermExpression: handle_monomialTermExpression_node, IndexedVar: handle_var_node, IndexTemplate: handle_indexTemplate_node, Numeric_GetItemExpression: handle_numericGetItemExpression_node, TemplateSumExpression: handle_templateSumExpression_node, ScalarParam: handle_param_node, _ParamData: handle_param_node, IndexedParam: handle_param_node, NPV_Numeric_GetItemExpression: handle_numericGetItemExpression_node, IndexedBlock: handle_indexedBlock_node, NPV_Structural_GetItemExpression: handle_npv_structuralGetItemExpression_node, str: handle_str_node, Numeric_GetAttrExpression: handle_numericGetAttrExpression_node, NPV_SumExpression: handle_sumExpression_node, NPV_DivisionExpression: handle_division_node}
        if numpy_available:
            self._operator_handles[np.float64] = handle_num_node

    def exitNode(self, node, data):
        try:
            return self._operator_handles[node.__class__](self, node, *data)
        except:
            raise DeveloperError('Latex printer encountered an error when processing type %s, contact the developers' % node.__class__)