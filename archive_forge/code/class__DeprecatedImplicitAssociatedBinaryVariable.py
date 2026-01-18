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
class _DeprecatedImplicitAssociatedBinaryVariable(object):
    __slots__ = ('_boolvar',)

    def __init__(self, boolvar):
        self._boolvar = weakref_ref(boolvar)

    def __call__(self):
        deprecation_warning('Relying on core.logical_to_linear to transform BooleanVars that do not appear in LogicalConstraints is deprecated. Please associate your own binaries if you have BooleanVars not used in logical expressions.', version='6.2')
        parent_block = self._boolvar().parent_block()
        new_var = Var(domain=Binary)
        parent_block.add_component(unique_component_name(parent_block, self._boolvar().local_name + '_asbinary'), new_var)
        self._boolvar()._associated_binary = None
        self._boolvar().associate_binary_var(new_var)
        return new_var

    def __getstate__(self):
        return self._boolvar()

    def __setstate__(self, state):
        self._boolvar = weakref_ref(state)