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
class _GetItemIndexer(object):

    def __init__(self, expr):
        self._base = expr.arg(0)
        self._args = []
        _hash = [id(self._base)]
        for x in expr.args[1:]:
            try:
                logging.disable(logging.CRITICAL)
                val = value(x)
                self._args.append(val)
                _hash.append(val)
            except TemplateExpressionError as e:
                if x is not e.template:
                    raise TypeError('Cannot use the param substituter with expression templates\nwhere the component index has the IndexTemplate in an expression.\n\tFound in %s' % (expr,))
                self._args.append(e.template)
                _hash.append(id(e.template._set))
            finally:
                logging.disable(logging.NOTSET)
        self._hash = tuple(_hash)

    def nargs(self):
        return len(self._args)

    def arg(self, i):
        return self._args[i]

    @property
    def base(self):
        return self._base

    @property
    def args(self):
        return self._args

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is _GetItemIndexer:
            return self._hash == other._hash
        else:
            return False

    def __str__(self):
        return '%s[%s]' % (self._base.name, ','.join((str(x) for x in self._args)))