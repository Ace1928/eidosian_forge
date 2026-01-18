import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
class OneClassSVM(OutlierMixin, BaseLibSVM):
    """Unsupervised Outlier Detection.

    Estimate the support of a high-dimensional distribution.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <outlier_detection>`.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,          default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (1, n_SV)
        Coefficients of the support vectors in the decision function.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (1,)
        Constant in the decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

        .. versionadded:: 1.1

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - `offset_`.
        The offset is the opposite of `intercept_` and is provided for
        consistency with other outlier detection algorithms.

        .. versionadded:: 0.20

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    See Also
    --------
    sklearn.linear_model.SGDOneClassSVM : Solves linear One-Class SVM using
        Stochastic Gradient Descent.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection using
        Local Outlier Factor (LOF).
    sklearn.ensemble.IsolationForest : Isolation Forest Algorithm.

    Examples
    --------
    >>> from sklearn.svm import OneClassSVM
    >>> X = [[0], [0.44], [0.45], [0.46], [1]]
    >>> clf = OneClassSVM(gamma='auto').fit(X)
    >>> clf.predict(X)
    array([-1,  1,  1,  1, -1])
    >>> clf.score_samples(X)
    array([1.7798..., 2.0547..., 2.0556..., 2.0561..., 1.7332...])
    """
    _impl = 'one_class'
    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints}
    for unused_param in ['C', 'class_weight', 'epsilon', 'probability', 'random_state']:
        _parameter_constraints.pop(unused_param)

    def __init__(self, *, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        super().__init__(kernel, degree, gamma, coef0, tol, 0.0, nu, 0.0, shrinking, False, cache_size, None, verbose, max_iter, random_state=None)

    def fit(self, X, y=None, sample_weight=None):
        """Detect the soft boundary of the set of samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Set of samples, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        If X is not a C-ordered contiguous array it is copied.
        """
        super().fit(X, np.ones(_num_samples(X)), sample_weight=sample_weight)
        self.offset_ = -self._intercept_
        return self

    def decision_function(self, X):
        """Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        dec : ndarray of shape (n_samples,)
            Returns the decision function of the samples.
        """
        dec = self._decision_function(X).ravel()
        return dec

    def score_samples(self, X):
        """Raw scoring function of the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        score_samples : ndarray of shape (n_samples,)
            Returns the (unshifted) scoring function of the samples.
        """
        return self.decision_function(X) + self.offset_

    def predict(self, X):
        """Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        y = super().predict(X)
        return np.asarray(y, dtype=np.intp)

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}