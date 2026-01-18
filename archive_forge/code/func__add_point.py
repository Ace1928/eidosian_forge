import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def _add_point(ds):
    sortds = list(ds)
    maxstep = sortds[1] - sortds[0]
    maxloc = 0
    for i in range(2, len(sortds)):
        if sortds[i] - sortds[i - 1] > maxstep:
            maxstep = sortds[i] - sortds[i - 1]
            maxloc = i - 1
    ds.add(round(sortds[maxloc] + maxstep / 2.0, 6))