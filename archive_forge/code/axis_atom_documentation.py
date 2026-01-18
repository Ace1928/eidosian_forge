import abc
from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A SciPy sparse matrix or None.
        