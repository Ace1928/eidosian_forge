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
class IndexTemplate(NumericValue):
    """A "placeholder" for an index value in template expressions.

    This class is a placeholder for an index value within a template
    expression.  That is, given the expression template for "m.x[i]",
    where `m.z` is indexed by `m.I`, the expression tree becomes:

    _GetItem:
       - m.x
       - IndexTemplate(_set=m.I, _value=None)

    Constructor Arguments:
       _set: the Set from which this IndexTemplate can take values
    """
    __slots__ = ('_set', '_value', '_index', '_id', '_group', '_lock')

    def __init__(self, _set, index=0, _id=None, _group=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id
        self._group = _group
        self._lock = None

    def __deepcopy__(self, memo):
        if '__block_scope__' in memo:
            memo[id(self)] = self
            return self
        return super().__deepcopy__(memo)

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is _NotSpecified:
            if exception:
                raise TemplateExpressionError(self, 'Evaluating uninitialized IndexTemplate (%s)' % (self,))
            return None
        else:
            return self._value

    def _resolve_template(self, args):
        assert not args
        return self()

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_potentially_variable(self):
        """Returns False because index values cannot be variables.

        The IndexTemplate represents a placeholder for an index value
        for an IndexedComponent, and at the moment, Pyomo does not
        support variable indirection.
        """
        return False

    def __str__(self):
        return self.getname()

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        if self._id is not None:
            return '_%s' % (self._id,)
        _set_name = self._set.getname(fully_qualified, name_buffer, relative_to)
        if self._index is not None and self._set.dimen != 1:
            _set_name += '(%s)' % (self._index,)
        return '{' + _set_name + '}'

    def set_value(self, values=_NotSpecified, lock=None):
        if lock is not self._lock:
            raise RuntimeError('The TemplateIndex %s is currently locked by %s and cannot be set through lock %s' % (self, self._lock, lock))
        if values is _NotSpecified:
            self._value = _NotSpecified
            return
        if type(values) is not tuple:
            values = (values,)
        if self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                self._value = values[self._index]
        else:
            self._value = values

    def lock(self, lock):
        assert self._lock is None
        self._lock = lock
        return self._value

    def unlock(self, lock):
        assert self._lock is lock
        self._lock = None