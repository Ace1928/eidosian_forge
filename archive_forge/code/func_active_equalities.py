from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def active_equalities(blk):
    """
    Generator returning active equality constraints in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Constraint, active=True):
        try:
            u = value(o.upper, exception=False)
            l = value(o.lower, exception=False)
            if u == l and l is not None:
                yield o
        except ZeroDivisionError:
            pass