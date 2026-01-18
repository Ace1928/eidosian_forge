from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _getAbstractBounds(comp):
    """
    Returns the bounds of this component
    """
    if getattr(comp, 'bounds', None) is None:
        return (None, None)
    else:
        return comp.bounds