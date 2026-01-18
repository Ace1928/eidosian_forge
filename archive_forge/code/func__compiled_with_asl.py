import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def _compiled_with_asl(self):
    results = subprocess.run([self.executable(), 'dummy', '-AMPL', '-stop'], timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return 'No match for AMPL'.lower() not in results.stdout.lower()