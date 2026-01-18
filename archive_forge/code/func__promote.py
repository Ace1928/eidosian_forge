import abc
from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
@staticmethod
def _promote(arg, shape: Tuple[int, ...]):
    """Promotes the lin op if necessary.

        Parameters
        ----------
        arg : LinOp
            LinOp to promote.
        shape : tuple
            The shape desired.

        Returns
        -------
        tuple
            Promoted LinOp.
        """
    if arg.shape != shape:
        return lu.promote(arg, shape)
    else:
        return arg