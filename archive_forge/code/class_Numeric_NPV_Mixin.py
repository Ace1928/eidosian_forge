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
class Numeric_NPV_Mixin(NPV_Mixin):
    __slots__ = ()

    def potentially_variable_base_class(self):
        cls = list(self.__class__.__bases__)
        cls.remove(Numeric_NPV_Mixin)
        assert len(cls) == 1
        return cls[0]

    def __neg__(self):
        return NPV_NegationExpression((self,))

    def __abs__(self):
        return NPV_AbsExpression((self,))