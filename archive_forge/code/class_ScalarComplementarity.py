from collections import namedtuple
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, native_numeric_types, as_numeric
from pyomo.core import Constraint, Var, Block, Set
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
import logging
class ScalarComplementarity(_ComplementarityData, Complementarity):

    def __init__(self, *args, **kwds):
        _ComplementarityData.__init__(self, self)
        Complementarity.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index