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
class LinearBeforeChildDispatcher(BeforeChildDispatcher):

    def __init__(self):
        self[ExternalFunctionExpression] = self._before_external
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _record_var(visitor, var):
        vm = visitor.var_map
        vo = visitor.var_order
        l = len(vo)
        try:
            _iter = var.parent_component().values(visitor.sorter)
        except AttributeError:
            _iter = (var,)
        for v in _iter:
            if v.fixed:
                continue
            vid = id(v)
            vm[vid] = v
            vo[vid] = l
            l += 1

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                return (False, (_CONSTANT, visitor.check_constant(child.value, child)))
            LinearBeforeChildDispatcher._record_var(visitor, child)
        ans = visitor.Result()
        ans.linear[_id] = 1
        return (False, (_LINEAR, ans))

    @staticmethod
    def _before_monomial(visitor, child):
        arg1, arg2 = child._args_
        if arg1.__class__ not in native_types:
            try:
                arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
            except (ValueError, ArithmeticError):
                return (True, None)
        _id = id(arg2)
        if _id not in visitor.var_map:
            if arg2.fixed:
                return (False, (_CONSTANT, arg1 * visitor.check_constant(arg2.value, arg2)))
            LinearBeforeChildDispatcher._record_var(visitor, arg2)
        if not arg1:
            if arg2.fixed:
                arg2 = visitor.check_constant(arg2.value, arg2)
                if arg2 != arg2:
                    deprecation_warning(f'Encountered {arg1}*{str(arg2.value)} in expression tree.  Mapping the NaN result to 0 for compatibility with the lp_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.6.0')
            return (False, (_CONSTANT, arg1))
        ans = visitor.Result()
        ans.linear[_id] = arg1
        return (False, (_LINEAR, ans))

    @staticmethod
    def _before_linear(visitor, child):
        var_map = visitor.var_map
        var_order = visitor.var_order
        ans = visitor.Result()
        const = 0
        linear = ans.linear
        for arg in child.args:
            if arg.__class__ is MonomialTermExpression:
                arg1, arg2 = arg._args_
                if arg1.__class__ not in native_types:
                    try:
                        arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
                    except (ValueError, ArithmeticError):
                        return (True, None)
                if not arg1:
                    if arg2.fixed:
                        arg2 = visitor.check_constant(arg2.value, arg2)
                        if arg2 != arg2:
                            deprecation_warning(f'Encountered {arg1}*{str(arg2.value)} in expression tree.  Mapping the NaN result to 0 for compatibility with the lp_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.6.0')
                    continue
                _id = id(arg2)
                if _id not in var_map:
                    if arg2.fixed:
                        const += arg1 * visitor.check_constant(arg2.value, arg2)
                        continue
                    LinearBeforeChildDispatcher._record_var(visitor, arg2)
                    linear[_id] = arg1
                elif _id in linear:
                    linear[_id] += arg1
                else:
                    linear[_id] = arg1
            elif arg.__class__ in native_numeric_types:
                const += arg
            else:
                try:
                    const += visitor.check_constant(visitor.evaluate(arg), arg)
                except (ValueError, ArithmeticError):
                    return (True, None)
        if linear:
            ans.constant = const
            return (False, (_LINEAR, ans))
        else:
            return (False, (_CONSTANT, const))

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            _type, expr = visitor.subexpression_cache[_id]
            if _type is _CONSTANT:
                return (False, (_type, expr))
            else:
                return (False, (_type, expr.duplicate()))
        else:
            return (True, None)

    @staticmethod
    def _before_external(visitor, child):
        ans = visitor.Result()
        if all((is_fixed(arg) for arg in child.args)):
            try:
                ans.constant = visitor.check_constant(visitor.evaluate(child), child)
                return (False, (_CONSTANT, ans))
            except:
                pass
        ans.nonlinear = child
        return (False, (_GENERAL, ans))