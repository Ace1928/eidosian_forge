import os
import subprocess
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
import logging

                This may occur if SCIP solves the problem during presolve. In that case,
                the log file may not get parsed correctly (self.read_scip_log), and
                results.solver.primal_bound will not be populated.
                