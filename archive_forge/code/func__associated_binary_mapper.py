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
def _associated_binary_mapper(encode, val):
    if val is None:
        return None
    if encode:
        if val.__class__ is not _DeprecatedImplicitAssociatedBinaryVariable:
            return val()
    elif val.__class__ is not _DeprecatedImplicitAssociatedBinaryVariable:
        return weakref_ref(val)
    return val