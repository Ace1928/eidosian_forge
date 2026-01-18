import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import (
from .metrics.pairwise import KERNEL_PARAMS, PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from .utils import check_random_state, deprecated
from .utils._param_validation import Interval, StrOptions
from .utils.extmath import safe_sparse_dot
from .utils.validation import (
class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for additive chi2 kernel.

    Uses sampling the fourier transform of the kernel characteristic
    at regular intervals.

    Since the kernel that is to be approximated is additive, the components of
    the input vectors can be treated separately.  Each entry in the original
    space is transformed into 2*sample_steps-1 features, where sample_steps is
    a parameter of the method. Typical values of sample_steps include 1, 2 and
    3.

    Optimal choices for the sampling interval for certain data ranges can be
    computed (see the reference). The default values should be reasonable.

    Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

    Parameters
    ----------
    sample_steps : int, default=2
        Gives the number of (complex) sampling points.

    sample_interval : float, default=None
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    Attributes
    ----------
    sample_interval_ : float
        Stored sampling interval. Specified as a parameter if `sample_steps`
        not in {1,2,3}.

        .. deprecated:: 1.3
           `sample_interval_` serves internal purposes only and will be removed in 1.5.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
        the chi squared kernel.

    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

    sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
        squared kernel.

    Notes
    -----
    This estimator approximates a slightly different version of the additive
    chi squared kernel then ``metric.additive_chi2`` computes.

    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
    2011

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.kernel_approximation import AdditiveChi2Sampler
    >>> X, y = load_digits(return_X_y=True)
    >>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    >>> X_transformed = chi2sampler.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
    >>> clf.fit(X_transformed, y)
    SGDClassifier(max_iter=5, random_state=0)
    >>> clf.score(X_transformed, y)
    0.9499...
    """
    _parameter_constraints: dict = {'sample_steps': [Interval(Integral, 1, None, closed='left')], 'sample_interval': [Interval(Real, 0, None, closed='left'), None]}

    def __init__(self, *, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs),                 default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = self._validate_data(X, accept_sparse='csr')
        check_non_negative(X, 'X in AdditiveChi2Sampler.fit')
        if self.sample_interval is None:
            if self.sample_steps == 1:
                self._sample_interval = 0.8
            elif self.sample_steps == 2:
                self._sample_interval = 0.5
            elif self.sample_steps == 3:
                self._sample_interval = 0.4
            else:
                raise ValueError('If sample_steps is not in [1, 2, 3], you need to provide sample_interval')
        else:
            self._sample_interval = self.sample_interval
        return self

    @deprecated('The ``sample_interval_`` attribute was deprecated in version 1.3 and will be removed 1.5.')
    @property
    def sample_interval_(self):
        return self._sample_interval

    def transform(self, X):
        """Apply approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : {ndarray, sparse matrix},                shape = (n_samples, n_features * (2*sample_steps - 1))
            Whether the return value is an array or sparse matrix depends on
            the type of the input X.
        """
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        check_non_negative(X, 'X in AdditiveChi2Sampler.transform')
        sparse = sp.issparse(X)
        if hasattr(self, '_sample_interval'):
            sample_interval = self._sample_interval
        elif self.sample_interval is None:
            if self.sample_steps == 1:
                sample_interval = 0.8
            elif self.sample_steps == 2:
                sample_interval = 0.5
            elif self.sample_steps == 3:
                sample_interval = 0.4
            else:
                raise ValueError('If sample_steps is not in [1, 2, 3], you need to provide sample_interval')
        else:
            sample_interval = self.sample_interval
        transf = self._transform_sparse if sparse else self._transform_dense
        return transf(X, self.sample_steps, sample_interval)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, 'n_features_in_')
        input_features = _check_feature_names_in(self, input_features, generate_names=True)
        est_name = self.__class__.__name__.lower()
        names_list = [f'{est_name}_{name}_sqrt' for name in input_features]
        for j in range(1, self.sample_steps):
            cos_names = [f'{est_name}_{name}_cos{j}' for name in input_features]
            sin_names = [f'{est_name}_{name}_sin{j}' for name in input_features]
            names_list.extend(cos_names + sin_names)
        return np.asarray(names_list, dtype=object)

    @staticmethod
    def _transform_dense(X, sample_steps, sample_interval):
        non_zero = X != 0.0
        X_nz = X[non_zero]
        X_step = np.zeros_like(X)
        X_step[non_zero] = np.sqrt(X_nz * sample_interval)
        X_new = [X_step]
        log_step_nz = sample_interval * np.log(X_nz)
        step_nz = 2 * X_nz * sample_interval
        for j in range(1, sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))
            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
            X_new.append(X_step)
            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
            X_new.append(X_step)
        return np.hstack(X_new)

    @staticmethod
    def _transform_sparse(X, sample_steps, sample_interval):
        indices = X.indices.copy()
        indptr = X.indptr.copy()
        data_step = np.sqrt(X.data * sample_interval)
        X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
        X_new = [X_step]
        log_step_nz = sample_interval * np.log(X.data)
        step_nz = 2 * X.data * sample_interval
        for j in range(1, sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))
            data_step = factor_nz * np.cos(j * log_step_nz)
            X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
            X_new.append(X_step)
            data_step = factor_nz * np.sin(j * log_step_nz)
            X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
            X_new.append(X_step)
        return sp.hstack(X_new)

    def _more_tags(self):
        return {'stateless': True, 'requires_positive_X': True}