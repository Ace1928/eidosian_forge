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
class DivisionExpression(NumericExpression):
    """
    Division expressions::

        x/y
    """
    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
        if result[1] == 0:
            return result[0]
        return None

    def getname(self, *args, **kwds):
        return 'div'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f'{self.getname()}({', '.join(values)})'
        return f'{values[0]}/{values[1]}'

    def _apply_operation(self, result):
        return result[0] / result[1]