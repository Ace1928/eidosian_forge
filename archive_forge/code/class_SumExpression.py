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
class SumExpression(NumericExpression):
    """
    Sum expression::

        x + y + ...

    This node represents an "n-ary" sum expression over at least 2 arguments.

    Args:
        args (list): Children nodes

    """
    __slots__ = ('_nargs',)
    PRECEDENCE = 6

    def __init__(self, args):
        if args.__class__ is not list:
            args = list(args)
        self._args_ = args
        self._nargs = len(args)

    def nargs(self):
        return self._nargs

    @property
    def args(self):
        return self._args_[:self._nargs]

    def getname(self, *args, **kwds):
        return 'sum'

    def _trunc_append(self, other):
        _args = self._args_
        if len(_args) > self._nargs:
            _args = _args[:self._nargs]
        _args.append(other)
        return self.__class__(_args)

    def _trunc_extend(self, other):
        _args = self._args_
        if len(_args) > self._nargs:
            _args = _args[:self._nargs]
        if len(other._args_) == other._nargs:
            _args.extend(other._args_)
        else:
            _args.extend(other._args_[:other._nargs])
        return self.__class__(_args)

    def _apply_operation(self, result):
        return sum(result)

    def _compute_polynomial_degree(self, result):
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans

    def _to_string(self, values, verbose, smap):
        if not values:
            values = ['0']
        if verbose:
            return f'{self.getname()}({', '.join(values)})'
        term = values[0]
        for i in range(1, len(values)):
            term = values[i]
            if term[0] in '-+':
                values[i] = term[0] + ' ' + term[1:].strip()
            else:
                values[i] = '+ ' + term.strip()
        return ' '.join(values)

    @deprecated("SumExpression.add() is deprecated.  Please use regular Python operators (infix '+' or inplace '+='.)", version='6.6.0')
    def add(self, new_arg):
        self += new_arg
        return self