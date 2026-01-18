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
class UnaryFunctionExpression(NumericExpression):
    """
    An expression object for intrinsic (math) functions (e.g. sin, cos, tan).

    Args:
        args (tuple): Children nodes
        name (string): The function name
        fcn: The function that is used to evaluate this expression
    """
    __slots__ = ('_fcn', '_name')
    PRECEDENCE = None

    def __init__(self, args, name=None, fcn=None):
        self._args_ = args
        self._name = name
        self._fcn = fcn

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._name, self._fcn)

    def getname(self, *args, **kwds):
        return self._name

    def _to_string(self, values, verbose, smap):
        return f'{self.getname()}({', '.join(values)})'

    def _apply_operation(self, result):
        return self._fcn(result[0])