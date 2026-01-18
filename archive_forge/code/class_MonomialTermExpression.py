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
class MonomialTermExpression(ProductExpression):
    __slots__ = ()

    def getname(self, *args, **kwds):
        return 'mon'

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            return operator.mul(*args)
        return self.__class__(args)