import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.expression import _ExpressionData, _GeneralExpressionDataImpl
from pyomo.core.base.set import Set
from pyomo.core.base.initializer import (
from pyomo.core.base import minimize, maximize
class SimpleObjective(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarObjective
    __renamed__version__ = '6.0'