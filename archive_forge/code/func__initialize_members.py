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
def _initialize_members(self, initSet):
    for idx in initSet:
        tmp = self[idx]
        for key in self._implicit:
            tmp.add(None, key)
        if self._extends:
            for key, val in self._extends.vars.items():
                tmp.add(val, key)
        for key, val in self._initialize.items():
            tmp.add(val, key)
        if self._rule:
            items = apply_indexed_rule(self, self._rule, self._parent(), idx)
            for key, val in items.items():
                tmp.add(val, key)