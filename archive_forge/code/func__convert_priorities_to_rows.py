import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentMap, Bunch
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver, BranchDirection
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.solvers.mockmip import MockMIP
from pyomo.core.base import Var, Suffix, active_export_suffix_generator
from pyomo.core.kernel.suffix import export_suffix_generator
from pyomo.core.kernel.block import IBlock
from pyomo.util.components import iter_component
def _convert_priorities_to_rows(self, instance, priorities, directions):
    if isinstance(instance, IBlock):
        smap = getattr(instance, '._symbol_maps')[self._smap_id]
    else:
        smap = instance.solutions.symbol_map[self._smap_id]
    byObject = smap.byObject
    rows = []
    for var, priority in priorities.items():
        if priority is None or not var.active:
            continue
        if not 0 <= priority == int(priority):
            raise ValueError('`priority` must be a non-negative integer')
        var_direction = directions.get(var, BranchDirection.default)
        for child_var in iter_component(var):
            if id(child_var) not in byObject:
                continue
            child_var_direction = directions.get(child_var, var_direction)
            rows.append((byObject[id(child_var)], priority, child_var_direction))
    return rows