import abc
import numpy as np
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.expressions import cvxtypes
def _chain_constraints(self):
    """Raises an error due to chained constraints.
        """
    raise Exception('Cannot evaluate the truth value of a constraint or chain constraints, e.g., 1 >= x >= 0.')