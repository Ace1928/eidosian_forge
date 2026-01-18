import logging
import sys
from pyomo.common.pyomo_typing import overload
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.core.expr.numvalue import (
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.core.base.units_container import units
class ScalarVar(_GeneralVarData, Var):
    """A single variable."""

    def __init__(self, *args, **kwd):
        _GeneralVarData.__init__(self, component=self)
        Var.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index