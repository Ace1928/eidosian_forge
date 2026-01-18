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
class AtMostExpression(NaryBooleanExpression):
    """
    Logical constraint that at most N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: atmost(1, True, False, False) --> True

    """
    __slots__ = ()
    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'atmost'

    def _to_string(self, values, verbose, smap):
        return 'atmost(%s: [%s])' % (values[0], ', '.join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) <= result[0]