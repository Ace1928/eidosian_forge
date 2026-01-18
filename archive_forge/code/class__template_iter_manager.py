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
class _template_iter_manager(object):

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

    class _pause_template_iter_manager(object):
        __slots__ = ('iter_manager',)

        def __init__(self, iter_manager):
            self.iter_manager = iter_manager

        def __enter__(self):
            self.iter_manager.release()
            return self

        def __exit__(self, et, ev, tb):
            self.iter_manager.acquire()

    def __init__(self):
        self.paused = True
        self.context = None
        self.iters = None
        self.builtin_sum = builtins.sum

    def init(self, context, *iter_fcns):
        assert self.context is None
        self.context = context
        self.iters = [self._iter_wrapper(it, context) for it in iter_fcns]
        return self

    def acquire(self):
        assert self.paused
        self.paused = False
        builtins.sum = self.context.sum_template
        for it in self.iters:
            it.acquire()

    def release(self):
        assert not self.paused
        self.paused = True
        builtins.sum = self.builtin_sum
        for it in self.iters:
            it.release()

    def __enter__(self):
        assert self.context
        self.acquire()
        return self

    def __exit__(self, et, ev, tb):
        self.release()
        self.context = None
        self.iters = None

    def pause(self):
        if self.paused:
            return nullcontext()
        else:
            return self._pause_template_iter_manager(self)