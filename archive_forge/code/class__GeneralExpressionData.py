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
class _GeneralExpressionData(_GeneralExpressionDataImpl, ComponentData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr        The expression owned by this data.

    Private class attributes:
        _component  The expression component.
    """
    __slots__ = ('_args_',)

    def __init__(self, expr=None, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET