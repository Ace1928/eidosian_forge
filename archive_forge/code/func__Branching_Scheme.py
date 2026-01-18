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
def _Branching_Scheme(self, n):
    """
        Branching scheme for LOG, requires a gray code
        """
    BIGL = 2 ** n
    S = range(1, n + 1)
    G = {k: v for k, v in enumerate(_GrayCode(n), start=1)}
    L = {s: [k + 1 for k in range(BIGL + 1) if (k == 0 or G[k][s - 1] == 1) and (k == BIGL or G[k + 1][s - 1] == 1)] for s in S}
    R = {s: [k + 1 for k in range(BIGL + 1) if (k == 0 or G[k][s - 1] == 0) and (k == BIGL or G[k + 1][s - 1] == 0)] for s in S}
    return (S, L, R)