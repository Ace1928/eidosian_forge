import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import RenamedClass
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.common.numeric_types import (
import pyomo.core.expr as EXPR
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.base.initializer import Initializer
class _GeneralExpressionDataImpl(_ExpressionData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr       The expression owned by this data.
    """
    __slots__ = ()

    def __init__(self, expr=None):
        self._args_ = (expr,)

    def create_node_with_local_data(self, values):
        """
        Construct a simple expression after constructing the
        contained expression.

        This class provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.
        """
        obj = ScalarExpression()
        obj.construct()
        obj._args_ = values
        return obj

    def set_value(self, expr):
        """Set the expression on this expression."""
        if expr is None or expr.__class__ in native_numeric_types:
            self._args_ = (expr,)
            return
        try:
            if expr.is_numeric_type():
                self._args_ = (expr,)
                return
        except AttributeError:
            if check_if_numeric_type(expr):
                self._args_ = (expr,)
                return
        raise ValueError(f"Cannot assign {expr.__class__.__name__} to '{self.name}': {self.__class__.__name__} components only allow numeric expression types.")

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        e, = self._args_
        return e.__class__ in native_types or e.is_fixed()

    def __iadd__(self, other):
        e, = self._args_
        return numeric_expr._add_dispatcher[e.__class__, other.__class__](e, other)

    def __imul__(self, other):
        e, = self._args_
        return numeric_expr._mul_dispatcher[e.__class__, other.__class__](e, other)

    def __idiv__(self, other):
        e, = self._args_
        return numeric_expr._div_dispatcher[e.__class__, other.__class__](e, other)

    def __itruediv__(self, other):
        e, = self._args_
        return numeric_expr._div_dispatcher[e.__class__, other.__class__](e, other)

    def __ipow__(self, other):
        e, = self._args_
        return numeric_expr._pow_dispatcher[e.__class__, other.__class__](e, other)