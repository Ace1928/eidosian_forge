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
class _template_iter_context(object):
    """Manage the iteration context when generating templatized rules

    This class manages the context tracking when generating templatized
    rules.  It has two methods (`sum_template` and `get_iter`) that
    replace standard functions / methods (`sum` and
    :py:meth:`_FiniteSetMixin.__iter__`, respectively).  It also tracks
    unique identifiers for IndexTemplate objects and their groupings
    within `sum()` generators.
    """

    def __init__(self):
        self.cache = []
        self._id = 0
        self._group = 0

    def get_iter(self, _set):
        return _set_iterator_template_generator(_set, self)

    def npop_cache(self, n):
        result = self.cache[-n:]
        self.cache[-n:] = []
        return result

    def next_id(self):
        self._id += 1
        return self._id

    def next_group(self):
        self._group += 1
        return self._group

    def sum_template(self, generator):
        init_cache = len(self.cache)
        expr = next(generator)
        final_cache = len(self.cache)
        return TemplateSumExpression((expr,), self.npop_cache(final_cache - init_cache))