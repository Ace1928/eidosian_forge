from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
@staticmethod
def _parse_special_values(value):
    if value == 1e+300 or value == 2e+300:
        return float('nan')
    if value == 3e+300:
        return float('inf')
    if value == 4e+300:
        return -float('inf')
    if value == 5e+300:
        return sys.float_info.epsilon
    return value