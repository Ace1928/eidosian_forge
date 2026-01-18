from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
import cvxpy.utilities.linalg as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import performance_utils as perf
def _compute_attr(self) -> None:
    """Compute the attributes of the constant related to complex/real, sign.
        """
    is_real, is_imag = intf.is_complex(self.value)
    if self.is_complex():
        is_nonneg = is_nonpos = False
    else:
        is_nonneg, is_nonpos = intf.sign(self.value)
    self._imag = is_imag and (not is_real)
    self._nonpos = is_nonpos
    self._nonneg = is_nonneg