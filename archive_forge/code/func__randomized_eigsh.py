import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
def _randomized_eigsh(M, n_components, *, n_oversamples=10, n_iter='auto', power_iteration_normalizer='auto', selection='module', random_state=None):
    """Computes a truncated eigendecomposition using randomized methods

    This method solves the fixed-rank approximation problem described in the
    Halko et al paper.

    The choice of which components to select can be tuned with the `selection`
    parameter.

    .. versionadded:: 0.24

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose, it should be real symmetric square or complex
        hermitian

    n_components : int
        Number of eigenvalues and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of eigenvectors and eigenvalues. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See Halko
        et al (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see Halko et al paper, page 9).

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    selection : {'value', 'module'}, default='module'
        Strategy used to select the n components. When `selection` is `'value'`
        (not yet implemented, will become the default when implemented), the
        components corresponding to the n largest eigenvalues are returned.
        When `selection` is `'module'`, the components corresponding to the n
        eigenvalues with largest modules are returned.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    eigendecomposition using randomized methods to speed up the computations.

    This method is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    Strategy 'value': not implemented yet.
    Algorithms 5.3, 5.4 and 5.5 in the Halko et al paper should provide good
    candidates for a future implementation.

    Strategy 'module':
    The principle is that for diagonalizable matrices, the singular values and
    eigenvalues are related: if t is an eigenvalue of A, then :math:`|t|` is a
    singular value of A. This method relies on a randomized SVD to find the n
    singular components corresponding to the n singular values with largest
    modules, and then uses the signs of the singular vectors to find the true
    sign of t: if the sign of left and right singular vectors are different
    then the corresponding eigenvalue is negative.

    Returns
    -------
    eigvals : 1D array of shape (n_components,) containing the `n_components`
        eigenvalues selected (see ``selection`` parameter).
    eigvecs : 2D array of shape (M.shape[0], n_components) containing the
        `n_components` eigenvectors corresponding to the `eigvals`, in the
        corresponding order. Note that this follows the `scipy.linalg.eigh`
        convention.

    See Also
    --------
    :func:`randomized_svd`

    References
    ----------
    * :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      (Algorithm 4.3 for strategy 'module') <0909.4061>`
      Halko, et al. (2009)
    """
    if selection == 'value':
        raise NotImplementedError()
    elif selection == 'module':
        U, S, Vt = randomized_svd(M, n_components=n_components, n_oversamples=n_oversamples, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer, flip_sign=False, random_state=random_state)
        eigvecs = U[:, :n_components]
        eigvals = S[:n_components]
        diag_VtU = np.einsum('ji,ij->j', Vt[:n_components, :], U[:, :n_components])
        signs = np.sign(diag_VtU)
        eigvals = eigvals * signs
    else:
        raise ValueError('Invalid `selection`: %r' % selection)
    return (eigvals, eigvecs)