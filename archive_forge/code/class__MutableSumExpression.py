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
class _MutableSumExpression(SumExpression):
    """
    A mutable SumExpression

    The :func:`add` method is slightly different in that it
    does not create a new sum expression, but modifies the
    :attr:`_args_` data in place.
    """
    __slots__ = ()

    def make_immutable(self):
        self.__class__ = SumExpression

    def __iadd__(self, other):
        return _iadd_mutablesum_dispatcher[other.__class__](self, other)