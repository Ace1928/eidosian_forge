import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _set_instance(self, model, kwds={}):
    self._range_constraints = set()
    super(MOSEKDirect, self)._set_instance(model, kwds)
    self._pyomo_cone_to_solver_cone_map = dict()
    self._solver_cone_to_pyomo_cone_map = ComponentMap()
    self._whichsol = getattr(mosek.soltype, kwds.pop('soltype', 'bas'))
    try:
        self._solver_model = self._mosek_env.Task()
    except:
        err_msg = sys.exc_info()[1]
        logger.error('MOSEK task creation failed. ' + 'Reason: {}'.format(err_msg))
        raise
    self._add_block(model)