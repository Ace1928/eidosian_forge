from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _partial_deepcopy(memo={}):
    return func