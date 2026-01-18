import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _compute_precision_cholesky_from_precisions(precisions, covariance_type):
    """Compute the Cholesky decomposition of precisions using precisions themselves.

    As implemented in :func:`_compute_precision_cholesky`, the `precisions_cholesky_` is
    an upper-triangular matrix for each Gaussian component, which can be expressed as
    the $UU^T$ factorization of the precision matrix for each Gaussian component, where
    $U$ is an upper-triangular matrix.

    In order to use the Cholesky decomposition to get $UU^T$, the precision matrix
    $\\Lambda$ needs to be permutated such that its rows and columns are reversed, which
    can be done by applying a similarity transformation with an exchange matrix $J$,
    where the 1 elements reside on the anti-diagonal and all other elements are 0. In
    particular, the Cholesky decomposition of the transformed precision matrix is
    $J\\Lambda J=LL^T$, where $L$ is a lower-triangular matrix. Because $\\Lambda=UU^T$
    and $J=J^{-1}=J^T$, the `precisions_cholesky_` for each Gaussian component can be
    expressed as $JLJ$.

    Refer to #26415 for details.

    Parameters
    ----------
    precisions : array-like
        The precision matrix of the current components.
        The shape depends on the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends on the covariance_type.
    """
    if covariance_type == 'full':
        precisions_cholesky = np.array([_flipudlr(linalg.cholesky(_flipudlr(precision), lower=True)) for precision in precisions])
    elif covariance_type == 'tied':
        precisions_cholesky = _flipudlr(linalg.cholesky(_flipudlr(precisions), lower=True))
    else:
        precisions_cholesky = np.sqrt(precisions)
    return precisions_cholesky