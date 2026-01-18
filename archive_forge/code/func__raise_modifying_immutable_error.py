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
def _raise_modifying_immutable_error(obj, index):
    if obj.is_indexed():
        name = '%s[%s]' % (obj.name, index)
    else:
        name = obj.name
    raise TypeError('Attempting to set the value of the immutable parameter %s after the parameter has been constructed.  If you intend to change the value of this parameter dynamically, please declare the parameter as mutable [i.e., Param(mutable=True)]' % (name,))