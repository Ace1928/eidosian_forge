import sys
import types
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression
class _ImplicitAny(_AnySet):
    """An Any that issues a deprecation warning for non-Real values.

    This is a helper class to implement the deprecation warnings for the
    change of Param's implicit domain from Any to Reals.

    """
    __slots__ = ('_owner',)
    __autoslot_mappers__ = {'_owner': AutoSlots.weakref_mapper}

    def __new__(cls, **kwargs):
        return super().__new__(cls)

    def __init__(self, owner, **kwargs):
        self._owner = weakref_ref(owner)
        super().__init__(**kwargs)
        self._component = weakref_ref(self)
        self.construct()
        object.__setattr__(self, '_parent', None)
        self._bounds = (None, None)
        self._interval = (None, None, None)

    def __contains__(self, val):
        if val not in Reals:
            if self._owner is None or self._owner() is None:
                name = 'Unknown'
            else:
                name = self._owner().name
            deprecation_warning(f"Param '{name}' declared with an implicit domain of 'Any'. The default domain for Param objects is 'Any'.  However, we will be changing that default to 'Reals' in the future.  If you really intend the domain of this Paramto be 'Any', you can suppress this warning by explicitly specifying 'within=Any' to the Param constructor.", version='5.6.9', remove_in='6.0')
        return True

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        return super().getname(False, name_buffer, relative_to)

    @property
    def _parent(self):
        if self._owner is None or self._owner() is None:
            return None
        return self._owner()._parent

    @_parent.setter
    def _parent(self, val):
        pass