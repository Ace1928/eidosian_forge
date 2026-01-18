import logging
import sys
from weakref import ref as weakref_ref
from pyomo.common.deprecation import deprecated, RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule
class _ConnectorData(ComponentData, NumericValue):
    """Holds the actual connector information"""
    __slots__ = ('vars', 'aggregators')

    def __init__(self, component=None):
        """Constructor"""
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self.vars = {}
        self.aggregators = {}

    def set_value(self, value):
        msg = "Cannot specify the value of a connector '%s'"
        raise ValueError(msg % self.name)

    def is_fixed(self):
        """Return True if all vars/expressions in the Connector are fixed"""
        return all((v.is_fixed() for v in self._iter_vars()))

    def is_constant(self):
        """Return False

        Because the expression generation logic will attempt to evaluate
        constant subexpressions, a Connector can never be constant.
        """
        return False

    def is_potentially_variable(self):
        """Return True as connectors may (should!) contain variables"""
        return True

    def polynomial_degree(self):
        ans = 0
        for v in self._iter_vars():
            tmp = v.polynomial_degree()
            if tmp is None:
                return None
            ans = max(ans, tmp)
        return ans

    def is_binary(self):
        return len(self) and all((v.is_binary() for v in self._iter_vars()))

    def is_integer(self):
        return len(self) and all((v.is_integer() for v in self._iter_vars()))

    def is_continuous(self):
        return len(self) and all((v.is_continuous() for v in self._iter_vars()))

    def add(self, var, name=None, aggregate=None):
        if name is None:
            name = var.local_name
        if name in self.vars:
            raise ValueError("Cannot insert duplicate variable name '%s' into Connector '%s'" % (name, self.name))
        self.vars[name] = var
        if aggregate is not None:
            self.aggregators[name] = aggregate

    def _iter_vars(self):
        for var in self.vars.values():
            if not hasattr(var, 'is_indexed') or not var.is_indexed():
                yield var
            else:
                for v in var.values():
                    yield v