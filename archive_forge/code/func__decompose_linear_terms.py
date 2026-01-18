import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
def _decompose_linear_terms(expr, multiplier=1):
    """
    A generator function that yields tuples for the linear terms
    in an expression.  If nonlinear terms are encountered, this function
    raises the :class:`LinearDecompositionError` exception.

    Args:
        expr (expression): The root node of an expression tree

    Yields:
        Tuples: ``(coef, value)``.  If :attr:`value` is :const:`None`,
        then this represents a constant term with value :attr:`coef`.
        Otherwise, :attr:`value` is a variable object, and :attr:`coef`
        is the numeric coefficient.

    Raises:
        :class:`LinearDecompositionError` if a nonlinear term is encountered.
    """
    if expr.__class__ in native_numeric_types or not expr.is_potentially_variable():
        yield (multiplier * expr, None)
    elif expr.is_variable_type():
        yield (multiplier, expr)
    elif expr.__class__ is MonomialTermExpression:
        yield (multiplier * expr._args_[0], expr._args_[1])
    elif expr.__class__ is ProductExpression:
        if expr._args_[0].__class__ in native_numeric_types or not expr._args_[0].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[1], multiplier * expr._args_[0])
        elif expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[0], multiplier * expr._args_[1])
        else:
            raise LinearDecompositionError('Quadratic terms exist in a product expression.')
    elif expr.__class__ is DivisionExpression:
        if expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[0], multiplier / expr._args_[1])
        else:
            raise LinearDecompositionError('Unexpected nonlinear term (division)')
    elif isinstance(expr, SumExpression):
        for arg in expr.args:
            yield from _decompose_linear_terms(arg, multiplier)
    elif expr.__class__ is NegationExpression:
        yield from _decompose_linear_terms(expr._args_[0], -multiplier)
    else:
        raise LinearDecompositionError('Unexpected nonlinear term')