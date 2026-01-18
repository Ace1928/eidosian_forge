import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
class KernelCenterer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Center an arbitrary kernel matrix :math:`K`.

    Let define a kernel :math:`K` such that:

    .. math::
        K(X, Y) = \\phi(X) . \\phi(Y)^{T}

    :math:`\\phi(X)` is a function mapping of rows of :math:`X` to a
    Hilbert space and :math:`K` is of shape `(n_samples, n_samples)`.

    This class allows to compute :math:`\\tilde{K}(X, Y)` such that:

    .. math::
        \\tilde{K(X, Y)} = \\tilde{\\phi}(X) . \\tilde{\\phi}(Y)^{T}

    :math:`\\tilde{\\phi}(X)` is the centered mapped data in the Hilbert
    space.

    `KernelCenterer` centers the features without explicitly computing the
    mapping :math:`\\phi(\\cdot)`. Working with centered kernels is sometime
    expected when dealing with algebra computation such as eigendecomposition
    for :class:`~sklearn.decomposition.KernelPCA` for instance.

    Read more in the :ref:`User Guide <kernel_centering>`.

    Attributes
    ----------
    K_fit_rows_ : ndarray of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.kernel_approximation.Nystroem : Approximate a kernel map
        using a subset of the training data.

    References
    ----------
    .. [1] `Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller.
       "Nonlinear component analysis as a kernel eigenvalue problem."
       Neural computation 10.5 (1998): 1299-1319.
       <https://www.mlpack.org/papers/kpca.pdf>`_

    Examples
    --------
    >>> from sklearn.preprocessing import KernelCenterer
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelCenterer().fit(K)
    >>> transformer
    KernelCenterer()
    >>> transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    """

    def __init__(self):
        pass

    def fit(self, K, y=None):
        """Fit KernelCenterer.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        xp, _ = get_namespace(K)
        K = self._validate_data(K, dtype=_array_api.supported_float_dtypes(xp))
        if K.shape[0] != K.shape[1]:
            raise ValueError('Kernel matrix must be a square matrix. Input is a {}x{} matrix.'.format(K.shape[0], K.shape[1]))
        n_samples = K.shape[0]
        self.K_fit_rows_ = xp.sum(K, axis=0) / n_samples
        self.K_fit_all_ = xp.sum(self.K_fit_rows_) / n_samples
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
            Returns the instance itself.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(K)
        K = self._validate_data(K, copy=copy, dtype=_array_api.supported_float_dtypes(xp), reset=False)
        K_pred_cols = (xp.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, None]
        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_
        return K

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.n_features_in_

    def _more_tags(self):
        return {'pairwise': True, 'array_api_support': True}