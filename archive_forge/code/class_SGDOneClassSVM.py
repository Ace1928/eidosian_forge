import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
class SGDOneClassSVM(BaseSGD, OutlierMixin):
    """Solves linear One-Class SVM using Stochastic Gradient Descent.

    This implementation is meant to be used with a kernel approximation
    technique (e.g. `sklearn.kernel_approximation.Nystroem`) to obtain results
    similar to `sklearn.svm.OneClassSVM` which uses a Gaussian kernel by
    default.

    Read more in the :ref:`User Guide <sgd_online_one_class_svm>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    nu : float, default=0.5
        The nu parameter of the One Class SVM: an upper bound on the
        fraction of training errors and a lower bound of the fraction of
        support vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`. Defaults to 1000.
        Values must be in the range `[1, inf)`.

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol). Defaults to 1e-3.
        Values must be in the range `[0.0, inf)`.

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.
        Defaults to True.

    verbose : int, default=0
        The verbosity level.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default='optimal'
        The learning rate schedule to use with `fit`. (If using `partial_fit`,
        learning rate must be controlled directly).

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.

    eta0 : float, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.
        Values must be in the range `[0.0, inf)`.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        Values must be in the range `(-inf, inf)`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So ``average=10`` will begin averaging after seeing 10
        samples.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features.

    offset_ : ndarray of shape (1,)
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - offset.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    loss_function_ : concrete ``LossFunction``

        .. deprecated:: 1.4
            ``loss_function_`` was deprecated in version 1.4 and will be removed in
            1.6.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.

    Notes
    -----
    This estimator has a linear complexity in the number of training samples
    and is thus better suited than the `sklearn.svm.OneClassSVM`
    implementation for datasets with a large number of training samples (say
    > 10,000).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> clf = linear_model.SGDOneClassSVM(random_state=42)
    >>> clf.fit(X)
    SGDOneClassSVM(random_state=42)

    >>> print(clf.predict([[4, 4]]))
    [1]
    """
    loss_functions = {'hinge': (Hinge, 1.0)}
    _parameter_constraints: dict = {**BaseSGD._parameter_constraints, 'nu': [Interval(Real, 0.0, 1.0, closed='right')], 'learning_rate': [StrOptions({'constant', 'optimal', 'invscaling', 'adaptive'}), Hidden(StrOptions({'pa1', 'pa2'}))], 'eta0': [Interval(Real, 0, None, closed='left')], 'power_t': [Interval(Real, None, None, closed='neither')]}

    def __init__(self, nu=0.5, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, warm_start=False, average=False):
        self.nu = nu
        super(SGDOneClassSVM, self).__init__(loss='hinge', penalty='l2', C=1.0, l1_ratio=0, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=DEFAULT_EPSILON, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=warm_start, average=average)

    def _fit_one_class(self, X, alpha, C, sample_weight, learning_rate, max_iter):
        """Uses SGD implementation with X and y=np.ones(n_samples)."""
        n_samples = X.shape[0]
        y = np.ones(n_samples, dtype=X.dtype, order='C')
        dataset, offset_decay = make_dataset(X, y, sample_weight)
        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)
        validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)
        validation_score_cb = self._make_validation_score_cb(validation_mask, X, y, sample_weight)
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        tol = self.tol if self.tol is not None else -np.inf
        one_class = 1
        pos_weight = 1
        neg_weight = 1
        if self.average:
            coef = self._standard_coef
            intercept = self._standard_intercept
            average_coef = self._average_coef
            average_intercept = self._average_intercept
        else:
            coef = self.coef_
            intercept = 1 - self.offset_
            average_coef = None
            average_intercept = [0]
        _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
        coef, intercept, average_coef, average_intercept, self.n_iter_ = _plain_sgd(coef, intercept[0], average_coef, average_intercept[0], self._loss_function_, penalty_type, alpha, C, self.l1_ratio, dataset, validation_mask, self.early_stopping, validation_score_cb, int(self.n_iter_no_change), max_iter, tol, int(self.fit_intercept), int(self.verbose), int(self.shuffle), seed, neg_weight, pos_weight, learning_rate_type, self.eta0, self.power_t, one_class, self.t_, offset_decay, self.average)
        self.t_ += self.n_iter_ * n_samples
        if self.average > 0:
            self._average_intercept = np.atleast_1d(average_intercept)
            self._standard_intercept = np.atleast_1d(intercept)
            if self.average <= self.t_ - 1.0:
                self.coef_ = average_coef
                self.offset_ = 1 - np.atleast_1d(average_intercept)
            else:
                self.coef_ = coef
                self.offset_ = 1 - np.atleast_1d(intercept)
        else:
            self.offset_ = 1 - np.atleast_1d(intercept)

    def _partial_fit(self, X, alpha, C, loss, learning_rate, max_iter, sample_weight, coef_init, offset_init):
        first_call = getattr(self, 'coef_', None) is None
        X = self._validate_data(X, None, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False, reset=first_call)
        n_features = X.shape[1]
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        if getattr(self, 'coef_', None) is None or coef_init is not None:
            self._allocate_parameter_mem(n_classes=1, n_features=n_features, input_dtype=X.dtype, coef_init=coef_init, intercept_init=offset_init, one_class=1)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError('Number of features %d does not match previous data %d.' % (n_features, self.coef_.shape[-1]))
        if self.average and getattr(self, '_average_coef', None) is None:
            self._average_coef = np.zeros(n_features, dtype=X.dtype, order='C')
            self._average_intercept = np.zeros(1, dtype=X.dtype, order='C')
        self._loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, 't_'):
            self.t_ = 1.0
        self._fit_one_class(X, alpha=alpha, C=C, learning_rate=learning_rate, sample_weight=sample_weight, max_iter=max_iter)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        """Fit linear One-Class SVM with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.
        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        if not hasattr(self, 'coef_'):
            self._more_validate_params(for_partial_fit=True)
        alpha = self.nu / 2
        return self._partial_fit(X, alpha, C=1.0, loss=self.loss, learning_rate=self.learning_rate, max_iter=1, sample_weight=sample_weight, coef_init=None, offset_init=None)

    def _fit(self, X, alpha, C, loss, learning_rate, coef_init=None, offset_init=None, sample_weight=None):
        if self.warm_start and hasattr(self, 'coef_'):
            if coef_init is None:
                coef_init = self.coef_
            if offset_init is None:
                offset_init = self.offset_
        else:
            self.coef_ = None
            self.offset_ = None
        self.t_ = 1.0
        self._partial_fit(X, alpha, C, loss, learning_rate, self.max_iter, sample_weight, coef_init, offset_init)
        if self.tol is not None and self.tol > -np.inf and (self.n_iter_ == self.max_iter):
            warnings.warn('Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.', ConvergenceWarning)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, coef_init=None, offset_init=None, sample_weight=None):
        """Fit linear One-Class SVM with Stochastic Gradient Descent.

        This solves an equivalent optimization problem of the
        One-Class SVM primal optimization problem and returns a weight vector
        w and an offset rho such that the decision function is given by
        <w, x> - rho.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        coef_init : array, shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization.

        offset_init : array, shape (n_classes,)
            The initial offset to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        self._more_validate_params()
        alpha = self.nu / 2
        self._fit(X, alpha=alpha, C=1.0, loss=self.loss, learning_rate=self.learning_rate, coef_init=coef_init, offset_init=offset_init, sample_weight=sample_weight)
        return self

    def decision_function(self, X):
        """Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an
        outlier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        dec : array-like, shape (n_samples,)
            Decision function values of the samples.
        """
        check_is_fitted(self, 'coef_')
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        decisions = safe_sparse_dot(X, self.coef_.T, dense_output=True) - self.offset_
        return decisions.ravel()

    def score_samples(self, X):
        """Raw scoring function of the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        score_samples : array-like, shape (n_samples,)
            Unshiffted scoring function values of the samples.
        """
        score_samples = self.decision_function(X) + self.offset_
        return score_samples

    def predict(self, X):
        """Return labels (1 inlier, -1 outlier) of the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        y : array, shape (n_samples,)
            Labels of the samples.
        """
        y = (self.decision_function(X) >= 0).astype(np.int32)
        y[y == 0] = -1
        return y

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}, 'preserves_dtype': [np.float64, np.float32]}