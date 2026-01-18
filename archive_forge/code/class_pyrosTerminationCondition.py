import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
class pyrosTerminationCondition(Enum):
    """Enumeration of all possible PyROS termination conditions."""
    robust_feasible = 0
    'Final solution is robust feasible.'
    robust_optimal = 1
    'Final solution is robust optimal.'
    robust_infeasible = 2
    'Problem is robust infeasible.'
    max_iter = 3
    'Maximum number of GRCS iteration reached.'
    subsolver_error = 4
    'Subsolver(s) provided could not solve a subproblem to\n    an acceptable termination status.'
    time_out = 5
    'Maximum allowable time exceeded.'

    @property
    def message(self):
        """
        str : Message associated with a given PyROS
        termination condition.
        """
        message_dict = {self.robust_optimal: 'Robust optimal solution identified.', self.robust_feasible: 'Robust feasible solution identified.', self.robust_infeasible: 'Problem is robust infeasible.', self.time_out: 'Maximum allowable time exceeded.', self.max_iter: 'Maximum number of iterations reached.', self.subsolver_error: 'Subordinate optimizer(s) could not solve a subproblem to an acceptable status.'}
        return message_dict[self]