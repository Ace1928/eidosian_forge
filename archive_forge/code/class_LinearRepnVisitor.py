import logging
import sys
from operator import itemgetter
from itertools import filterfalse
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
class LinearRepnVisitor(StreamBasedExpressionVisitor):
    Result = LinearRepn
    exit_node_handlers = _exit_node_handlers
    exit_node_dispatcher = ExitNodeDispatcher(_initialize_exit_node_dispatcher(_exit_node_handlers))
    expand_nonlinear_products = False
    max_exponential_expansion = 1

    def __init__(self, subexpression_cache, var_map, var_order, sorter):
        super().__init__()
        self.subexpression_cache = subexpression_cache
        self.var_map = var_map
        self.var_order = var_order
        self.sorter = sorter
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack

    def check_constant(self, ans, obj):
        if ans.__class__ not in native_numeric_types:
            if ans is None:
                return InvalidNumber(None, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
        if ans != ans:
            return InvalidNumber(nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'")
        return ans

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return (False, self.finalizeResult(result))
        return (True, expr)

    def beforeChild(self, node, child, child_idx):
        return _before_child_dispatcher[child.__class__](self, child)

    def enterNode(self, node):
        if node.__class__ in sum_like_expression_types:
            return (node.args, self.Result())
        else:
            return (node.args, [])

    def exitNode(self, node, data):
        if data.__class__ is self.Result:
            return data.walker_exitNode()
        return self.exit_node_dispatcher[node.__class__, *map(itemgetter(0), data)](self, node, *data)

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is self.Result:
            mult = ans.multiplier
            if mult == 1:
                zeros = list(filterfalse(itemgetter(1), ans.linear.items()))
                for vid, coef in zeros:
                    del ans.linear[vid]
            elif not mult:
                if ans.constant != ans.constant or any((c != c for c in ans.linear.values())):
                    deprecation_warning(f'Encountered {str(mult)}*nan in expression tree.  Mapping the NaN result to 0 for compatibility with the lp_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.6.0')
                return self.Result()
            else:
                linear = ans.linear
                zeros = []
                for vid, coef in linear.items():
                    if coef:
                        linear[vid] = coef * mult
                    else:
                        zeros.append(vid)
                for vid in zeros:
                    del linear[vid]
                if ans.nonlinear is not None:
                    ans.nonlinear *= mult
                if ans.constant:
                    ans.constant *= mult
                ans.multiplier = 1
            return ans
        ans = self.Result()
        assert result[0] is _CONSTANT
        ans.constant = result[1]
        return ans