from pyomo.opt.base.solvers import LegacySolverFactory
from pyomo.common.factory import Factory
from pyomo.contrib.solver.base import LegacySolverWrapper
class LegacySolver(LegacySolverWrapper, cls):
    pass