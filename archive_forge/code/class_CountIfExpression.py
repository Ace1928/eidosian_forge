import types
from itertools import islice
import logging
import traceback
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
from .numvalue import (
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType
from .numeric_expr import NumericExpression
import operator
class CountIfExpression(NumericExpression):
    """
    Logical expression that returns the number of True child statements.
    All arguments are expected to be Boolean-valued.
    """
    __slots__ = ()
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def getname(self, *arg, **kwd):
        return 'count_if'

    def _to_string(self, values, verbose, smap):
        return 'count_if(%s)' % ', '.join(values)

    def _apply_operation(self, result):
        return sum((value(r) for r in result))