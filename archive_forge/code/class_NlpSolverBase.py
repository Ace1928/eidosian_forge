from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
class NlpSolverBase(object):
    """A base class that solves an NLP object

    Subclasses should implement this interface for compatibility with
    ImplicitFunctionSolver objects.

    """

    def __init__(self, nlp, options=None, timer=None):
        raise NotImplementedError('%s has not implemented the __init__ method' % type(self))

    def solve(self, **kwds):
        raise NotImplementedError('%s has not implemented the solve method' % type(self))