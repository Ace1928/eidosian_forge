from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
Construct C++ LinOp tree from Python LinOp tree.

    Constructed C++ linOps are stored in the linPy_to_linC dict,
    which maps Python linOps to their corresponding C++ linOps.

    Parameters
    ----------
        linPy_to_linC: a dict for memoizing construction and storing
            the C++ LinOps
    