from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants.parameter import is_param_affine
@staticmethod
def _get_args_from_values(values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x = values[0].flatten()
    w = values[1].flatten()
    w_padded = np.zeros_like(x)
    w_padded[:len(w)] = w
    return (x, w_padded)