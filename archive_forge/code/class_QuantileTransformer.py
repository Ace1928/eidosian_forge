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
class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    For example visualizations, refer to :ref:`Compare QuantileTransformer with
    other scalers <plot_all_scaling_quantile_transformer_section>`.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, default=10_000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray of shape (n_quantiles, )
        Quantiles of references.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X)
    array([...])
    """
    _parameter_constraints: dict = {'n_quantiles': [Interval(Integral, 1, None, closed='left')], 'output_distribution': [StrOptions({'uniform', 'normal'})], 'ignore_implicit_zeros': ['boolean'], 'subsample': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state'], 'copy': ['boolean']}

    def __init__(self, *, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, subsample=10000, random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn("'ignore_implicit_zeros' takes effect only with sparse matrix. This parameter has no effect.")
        n_samples, n_features = X.shape
        references = self.references_ * 100
        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(n_samples, size=self.subsample, replace=False)
                col = col.take(subsample_idx, mode='clip')
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative. If a sparse matrix is provided,
            it will be converted into a sparse ``csc_matrix``.
        """
        n_samples, n_features = X.shape
        references = self.references_ * 100
        self.quantiles_ = []
        for feature_idx in range(n_features):
            column_nnz_data = X.data[X.indptr[feature_idx]:X.indptr[feature_idx + 1]]
            if len(column_nnz_data) > self.subsample:
                column_subsample = self.subsample * len(column_nnz_data) // n_samples
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample, dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                column_data[:column_subsample] = random_state.choice(column_nnz_data, size=column_subsample, replace=False)
            else:
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data), dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                column_data[:len(column_nnz_data)] = column_nnz_data
            if not column_data.size:
                self.quantiles_.append([0] * len(references))
            else:
                self.quantiles_.append(np.nanpercentile(column_data, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        if self.n_quantiles > self.subsample:
            raise ValueError('The number of quantiles cannot be greater than the number of samples used. Got {} quantiles and {} samples.'.format(self.n_quantiles, self.subsample))
        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]
        if self.n_quantiles > n_samples:
            warnings.warn('n_quantiles (%s) is greater than the total number of samples (%s). n_quantiles is set to n_samples.' % (self.n_quantiles, n_samples))
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
        rng = check_random_state(self.random_state)
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)
        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""
        output_distribution = self.output_distribution
        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            with np.errstate(invalid='ignore'):
                if output_distribution == 'normal':
                    X_col = stats.norm.cdf(X_col)
        with np.errstate(invalid='ignore'):
            if output_distribution == 'normal':
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == 'uniform':
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x
        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            X_col[isfinite_mask] = 0.5 * (np.interp(X_col_finite, quantiles, self.references_) - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1]))
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)
        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        if not inverse:
            with np.errstate(invalid='ignore'):
                if output_distribution == 'normal':
                    X_col = stats.norm.ppf(X_col)
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(X, reset=in_fit, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        with np.errstate(invalid='ignore'):
            if not accept_sparse_negative and (not self.ignore_implicit_zeros) and (sparse.issparse(X) and np.any(X.data < 0)):
                raise ValueError('QuantileTransformer only accepts non-negative sparse matrices.')
        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx], X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(X.data[column_slice], self.quantiles_[:, feature_idx], inverse)
        else:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(X[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)
        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, accept_sparse_negative=True, copy=self.copy)
        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {'allow_nan': True}