import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _get_te_dmatrix(design_matrices, constraints=None):
    """Builds tensor product design matrix, given the marginal design matrices.

    :param design_matrices: A sequence of 2-d arrays (marginal design matrices).
    :param constraints: The 2-d array defining model parameters (``betas``)
     constraints (``np.dot(constraints, betas) = 0``).
    :return: The (2-d array) design matrix.
    """
    dm = _row_tensor_product(design_matrices)
    if constraints is not None:
        dm = _absorb_constraints(dm, constraints)
    return dm