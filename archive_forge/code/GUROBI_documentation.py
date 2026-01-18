import os
import sys
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.core.kernel.block import IBlock
from pyomo.core import ConcreteModel, Var, Objective
from .gurobi_direct import gurobipy_available
from .ASL import ASL

        Returns a tuple describing the solver executable version.
        