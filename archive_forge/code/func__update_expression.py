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
def _update_expression(expre):
    """
    This method will construct any additional indices in an expression
    resulting from the discretization of a ContinuousSet.
    """
    expre.to_dense_data()