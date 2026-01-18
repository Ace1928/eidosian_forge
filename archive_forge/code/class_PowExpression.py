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
class PowExpression(NumericExpression):
    """
    Power expressions::

        x**y
    """
    __slots__ = ()
    PRECEDENCE = 2
    ASSOCIATIVITY = OperatorAssociativity.NON_ASSOCIATIVE

    def _compute_polynomial_degree(self, result):
        l, r = result
        if r == 0:
            exp = value(self._args_[1], exception=False)
            if exp is None:
                return None
            if exp == int(exp):
                if not exp:
                    return 0
                if l is not None and exp > 0:
                    return l * int(exp)
        return None

    def _is_fixed(self, args):
        if not args[1]:
            return False
        return args[0] or value(self._args_[1], exception=False) == 0

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f'{self.getname()}({', '.join(values)})'
        return f'{values[0]}**{values[1]}'