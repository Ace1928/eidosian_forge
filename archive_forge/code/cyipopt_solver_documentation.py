import io
import sys
import logging
import os
import abc
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.common.tee import redirect_fd, TeeStream
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition, ProblemSense
from pyomo.opt.results.solution import Solution
Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to
        the abstract class CyIpoptProblemInterface

        options can be provided as a dictionary of key value
        pairs
        