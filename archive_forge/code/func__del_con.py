from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import logging
def _del_con(c):
    parent = c.parent_component()
    if parent.is_indexed():
        parent.__delitem__(c.index())
    else:
        assert parent is c
        c.parent_block().del_component(c)