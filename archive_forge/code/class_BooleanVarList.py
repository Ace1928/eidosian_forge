import logging
from weakref import ref as weakref_ref, ReferenceType
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set, BooleanSet, Binary
from pyomo.core.base.util import is_functor
from pyomo.core.base.var import Var
@ModelComponentFactory.register('List of logical decision variables.')
class BooleanVarList(IndexedBooleanVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwargs):
        self._starting_index = kwargs.pop('starting_index', 1)
        args = (Set(),)
        IndexedBooleanVar.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """Construct this component."""
        if is_debug_set(logger):
            logger.debug('Constructing variable list %s', self.name)
        if self._value_init_value.__class__ is dict:
            for i in range(len(self._value_init_value)):
                self._index_set.add(i + self._starting_index)
        super(BooleanVarList, self).construct(data)
        if self._value_init_value.__class__ is dict:
            for k, v in self._value_init_value.items():
                self[k] = v

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        return self[next_idx]