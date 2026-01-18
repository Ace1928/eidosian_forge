import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from ..base import _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Interval
from ..utils.extmath import fast_logdet
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance
class MinCovDet(EmpiricalCovariance):
    """Minimum Covariance Determinant (MCD): robust estimator of covariance.

    The Minimum Covariance Determinant covariance estimator is to be applied
    on Gaussian-distributed data, but could still be relevant on data
    drawn from a unimodal, symmetric distribution. It is not meant to be used
    with multi-modal data (the algorithm used to fit a MinCovDet object is
    likely to fail in such a case).
    One should consider projection pursuit methods to deal with multi-modal
    datasets.

    Read more in the :ref:`User Guide <robust_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, the support of the robust location and the covariance
        estimates is computed, and a covariance estimate is recomputed from
        it, without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.

    support_fraction : float, default=None
        The proportion of points to be included in the support of the raw
        MCD estimate. Default is None, which implies that the minimum
        value of support_fraction will be used within the algorithm:
        `(n_samples + n_features + 1) / 2 * n_samples`. The parameter must be
        in the range (0, 1].

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    raw_location_ : ndarray of shape (n_features,)
        The raw robust estimated location before correction and re-weighting.

    raw_covariance_ : ndarray of shape (n_features, n_features)
        The raw robust estimated covariance before correction and re-weighting.

    raw_support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the raw robust estimates of location and shape, before correction
        and re-weighting.

    location_ : ndarray of shape (n_features,)
        Estimated robust location.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated robust covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    support_ : ndarray of shape (n_samples,)
        A mask of the observations that have been used to compute
        the robust estimates of location and shape.

    dist_ : ndarray of shape (n_samples,)
        Mahalanobis distances of the training set (on which :meth:`fit` is
        called) observations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    LedoitWolf : LedoitWolf Estimator.
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    References
    ----------

    .. [Rouseeuw1984] P. J. Rousseeuw. Least median of squares regression.
        J. Am Stat Ass, 79:871, 1984.
    .. [Rousseeuw] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    .. [ButlerDavies] R. W. Butler, P. L. Davies and M. Jhun,
        Asymptotics For The Minimum Covariance Determinant Estimator,
        The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import MinCovDet
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=500)
    >>> cov = MinCovDet(random_state=0).fit(X)
    >>> cov.covariance_
    array([[0.7411..., 0.2535...],
           [0.2535..., 0.3053...]])
    >>> cov.location_
    array([0.0813... , 0.0427...])
    """
    _parameter_constraints: dict = {**EmpiricalCovariance._parameter_constraints, 'support_fraction': [Interval(Real, 0, 1, closed='right'), None], 'random_state': ['random_state']}
    _nonrobust_covariance = staticmethod(empirical_covariance)

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, random_state=None):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit a Minimum Covariance Determinant with the FastMCD algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, ensure_min_samples=2, estimator='MinCovDet')
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        if (linalg.svdvals(np.dot(X.T, X)) > 1e-08).sum() != n_features:
            warnings.warn('The covariance matrix associated to your dataset is not full rank')
        raw_location, raw_covariance, raw_support, raw_dist = fast_mcd(X, support_fraction=self.support_fraction, cov_computation_method=self._nonrobust_covariance, random_state=random_state)
        if self.assume_centered:
            raw_location = np.zeros(n_features)
            raw_covariance = self._nonrobust_covariance(X[raw_support], assume_centered=True)
            precision = linalg.pinvh(raw_covariance)
            raw_dist = np.sum(np.dot(X, precision) * X, 1)
        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        self.dist_ = raw_dist
        self.correct_covariance(X)
        self.reweight_covariance(X)
        return self

    def correct_covariance(self, data):
        """Apply a correction to raw Minimum Covariance Determinant estimates.

        Correction using the empirical correction factor suggested
        by Rousseeuw and Van Driessen in [RVD]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        covariance_corrected : ndarray of shape (n_features, n_features)
            Corrected robust covariance estimate.

        References
        ----------

        .. [RVD] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """
        n_samples = len(self.dist_)
        n_support = np.sum(self.support_)
        if n_support < n_samples and np.allclose(self.raw_covariance_, 0):
            raise ValueError('The covariance matrix of the support data is equal to 0, try to increase support_fraction')
        correction = np.median(self.dist_) / chi2(data.shape[1]).isf(0.5)
        covariance_corrected = self.raw_covariance_ * correction
        self.dist_ /= correction
        return covariance_corrected

    def reweight_covariance(self, data):
        """Re-weight raw Minimum Covariance Determinant estimates.

        Re-weight observations using Rousseeuw's method (equivalent to
        deleting outlying observations from the data set before
        computing location and covariance estimates) described
        in [RVDriessen]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        location_reweighted : ndarray of shape (n_features,)
            Re-weighted robust location estimate.

        covariance_reweighted : ndarray of shape (n_features, n_features)
            Re-weighted robust covariance estimate.

        support_reweighted : ndarray of shape (n_samples,), dtype=bool
            A mask of the observations that have been used to compute
            the re-weighted robust location and covariance estimates.

        References
        ----------

        .. [RVDriessen] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """
        n_samples, n_features = data.shape
        mask = self.dist_ < chi2(n_features).isf(0.025)
        if self.assume_centered:
            location_reweighted = np.zeros(n_features)
        else:
            location_reweighted = data[mask].mean(0)
        covariance_reweighted = self._nonrobust_covariance(data[mask], assume_centered=self.assume_centered)
        support_reweighted = np.zeros(n_samples, dtype=bool)
        support_reweighted[mask] = True
        self._set_covariance(covariance_reweighted)
        self.location_ = location_reweighted
        self.support_ = support_reweighted
        X_centered = data - self.location_
        self.dist_ = np.sum(np.dot(X_centered, self.get_precision()) * X_centered, 1)
        return (location_reweighted, covariance_reweighted, support_reweighted)