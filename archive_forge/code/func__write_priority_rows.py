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
def _write_priority_rows(self, rows):
    with open(self._priorities_file_name, 'w') as ord_file:
        ord_file.write(ORDFileSchema.HEADER)
        for var_name, priority, direction in rows:
            ord_file.write(ORDFileSchema.ROW(var_name, priority, direction))
        ord_file.write(ORDFileSchema.FOOTER)