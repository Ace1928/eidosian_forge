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
class IndexedExpression(Expression):

    def add(self, index, expr):
        """Add an expression with a given index."""
        if type(expr) is tuple and expr == Expression.Skip:
            return None
        cdata = _GeneralExpressionData(expr, component=self)
        self._data[index] = cdata
        return cdata