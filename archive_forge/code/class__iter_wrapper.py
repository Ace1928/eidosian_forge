import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
class _iter_wrapper(object):
    __slots__ = ('_class', '_iter', '_old_iter')

    def __init__(self, cls, context):

        def _iter_fcn(obj):
            return context.get_iter(obj)
        self._class = cls
        self._old_iter = cls.__iter__
        self._iter = _iter_fcn

    def acquire(self):
        self._class.__iter__ = self._iter

    def release(self):
        self._class.__iter__ = self._old_iter