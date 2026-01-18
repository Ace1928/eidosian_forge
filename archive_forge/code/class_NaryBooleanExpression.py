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
class NaryBooleanExpression(BooleanExpression):
    """
    The abstract class for NaryBooleanExpression.

    This class should never be initialized.
    """
    __slots__ = ('_nargs',)

    def __init__(self, args):
        self._args_ = args
        self._nargs = len(self._args_)

    def nargs(self):
        """
        Return the number of expression arguments
        """
        return self._nargs

    def getname(self, *arg, **kwd):
        return 'NaryBooleanExpression'