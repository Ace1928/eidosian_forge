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
def _set_model(self, model):
    self._model = model