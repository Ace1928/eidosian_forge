from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _getAbstractIndices(comp):
    """
    Returns the index or index set of this component
    """
    if type(comp.index_set()) != type({}):
        return comp.index_set()
    else:
        return {None: None}