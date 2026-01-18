import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def _log_solver_intro_message(self):
    self.config.logger.info('Starting MindtPy version %s using %s algorithm' % ('.'.join(map(str, self.version())), self.config.strategy))
    os = StringIO()
    self.config.display(ostream=os)
    self.config.logger.info(os.getvalue())
    self.config.logger.info('-----------------------------------------------------------------------------------------------\n               Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)                \n-----------------------------------------------------------------------------------------------\nFor more information, please visit \nhttps://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html')
    self.config.logger.info('If you use this software, please cite the following:\nBernal, David E., et al. Mixed-integer nonlinear decomposition toolbox for Pyomo (MindtPy).\nComputer Aided Chemical Engineering. Vol. 44. Elsevier, 2018. 895-900.\n')