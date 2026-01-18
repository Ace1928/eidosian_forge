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
class MaxExpression(NumericExpression):
    """
    Maximum expressions::

        max(x, y, ...)
    """
    __slots__ = ()
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def _apply_operation(self, result):
        return max(result)

    def getname(self, *args, **kwds):
        return 'max'

    def _to_string(self, values, verbose, smap):
        return f'{self.getname()}({', '.join(values)})'