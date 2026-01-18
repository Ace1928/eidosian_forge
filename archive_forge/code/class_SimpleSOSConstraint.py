import sys
import logging
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.set_types import PositiveIntegers
class SimpleSOSConstraint(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarSOSConstraint
    __renamed__version__ = '6.0'