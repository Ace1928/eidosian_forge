from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
import cvxpy.utilities.linalg as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import performance_utils as perf
def _compute_symm_attr(self) -> None:
    """Determine whether the constant is symmetric/Hermitian.
        """
    is_symm, is_herm = intf.is_hermitian(self.value)
    self._symm = is_symm
    self._herm = is_herm