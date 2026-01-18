from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def _complete_index(loc, index, *newvals):
    """
    Function for inserting new values into a partial index.
    Used by get_index_set_except function to construct the
    index_getter function for completing indices of a particular
    component with particular sets excluded.

    Args:
        loc : Dictionary mapping location in the new index to
              location in newvals
        index : Partial index
        newvals : New values to insert into index. Can be scalars
                  or tuples (for higher-dimension sets)

    Returns:
        An index (tuple) with values from newvals inserted in
        locations specified by loc
    """
    if type(index) is not tuple:
        index = (index,)
    keys = sorted(loc.keys())
    if len(keys) != len(newvals):
        raise ValueError('Wrong number of values to complete index')
    for i in sorted(loc.keys()):
        newval = newvals[loc[i]]
        if type(newval) is not tuple:
            newval = (newval,)
        index = index[0:i] + newval + index[i:]
    return index