import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
class PWRepn(str, enum.Enum):
    SOS2 = 'SOS2'
    BIGM_BIN = 'BIGM_BIN'
    BIGM_SOS1 = 'BIGM_SOS1'
    CC = 'CC'
    DCC = 'DCC'
    DLOG = 'DLOG'
    LOG = 'LOG'
    MC = 'MC'
    INC = 'INC'