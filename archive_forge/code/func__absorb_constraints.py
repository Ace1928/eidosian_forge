import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _absorb_constraints(design_matrix, constraints):
    """Absorb model parameters constraints into the design matrix.

    :param design_matrix: The (2-d array) initial design matrix.
    :param constraints: The 2-d array defining initial model parameters
     (``betas``) constraints (``np.dot(constraints, betas) = 0``).
    :return: The new design matrix with absorbed parameters constraints.

    :raise ImportError: if scipy is not found, used for ``scipy.linalg.qr()``
      which is cleaner than numpy's version requiring a call like
      ``qr(..., mode='complete')`` to get a full QR decomposition.
    """
    try:
        from scipy import linalg
    except ImportError:
        raise ImportError('Cubic spline functionality requires scipy.')
    m = constraints.shape[0]
    q, r = linalg.qr(np.transpose(constraints))
    return np.dot(design_matrix, q[:, m:])