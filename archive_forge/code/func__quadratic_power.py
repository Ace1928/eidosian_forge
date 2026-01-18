from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.utilities as u
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import is_power2, pow_high, pow_mid, pow_neg
def _quadratic_power(self) -> bool:
    """Utility function to check if power is 0, 1 or 2."""
    p = self.p_rational
    return p in [0, 1, 2]