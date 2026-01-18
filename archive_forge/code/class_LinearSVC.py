import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
class LinearSVC(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVC` and
    :class:`~sklearn.svm.SVC` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default='l2'
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to ``coef_``
        vectors that are sparse.

    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss. The combination of ``penalty='l1'``
        and ``loss='hinge'`` is not supported.

    dual : "auto" or bool, default=True
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.
        `dual="auto"` will choose the value of the parameter automatically,
        based on the values of `n_samples`, `n_features`, `loss`, `multi_class`
        and `penalty`. If `n_samples` < `n_features` and optimizer supports
        chosen `loss`, `multi_class` and `penalty`, then dual will be set to True,
        otherwise it will be set to False.

        .. versionchanged:: 1.3
           The `"auto"` option is added in version 1.3 and will be the default
           in version 1.5.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        Determines the multi-class strategy if `y` contains more than
        two classes.
        ``"ovr"`` trains n_classes one-vs-rest classifiers, while
        ``"crammer_singer"`` optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from a theoretical perspective
        as it is consistent, it is seldom used in practice as it rarely leads
        to better accuracy and is more expensive to compute.
        If ``"crammer_singer"`` is chosen, the options loss, penalty and dual
        will be ignored.

    fit_intercept : bool, default=True
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: `[x_1, ..., x_n, 1]`, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float, default=1.0
        When `fit_intercept` is True, the instance vector x becomes ``[x_1,
        ..., x_n, intercept_scaling]``, i.e. a "synthetic" feature with a
        constant value equal to `intercept_scaling` is appended to the instance
        vector. The intercept becomes intercept_scaling * synthetic feature
        weight. Note that liblinear internally penalizes the intercept,
        treating it like any other term in the feature vector. To reduce the
        impact of the regularization on the intercept, the `intercept_scaling`
        parameter can be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to ``class_weight[i]*C`` for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        the dual coordinate descent (if ``dual=True``). When ``dual=False`` the
        underlying implementation of :class:`LinearSVC` is not random and
        ``random_state`` has no effect on the results.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2             else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal
        problem).

        ``coef_`` is a readonly property derived from ``raw_coef_`` that
        follows the internal memory layout of liblinear.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Maximum number of iterations run across all classes.

    See Also
    --------
    SVC : Implementation of Support Vector Machine classifier using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

        Furthermore SVC multi-class mode is implemented using one
        vs one scheme while LinearSVC uses one vs the rest. It is
        possible to implement one vs the rest with SVC by using the
        :class:`~sklearn.multiclass.OneVsRestClassifier` wrapper.

        Finally SVC can fit dense data without memory copy if the input
        is C-contiguous. Sparse data will still incur memory copy though.

    sklearn.linear_model.SGDClassifier : SGDClassifier can optimize the same
        cost function as LinearSVC
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon
    to have slightly different results for the same input data. If
    that happens, try with a smaller ``tol`` parameter.

    The underlying implementation, liblinear, uses a sparse internal
    representation for the data that will incur a memory copy.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------
    `LIBLINEAR: A Library for Large Linear Classification
    <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = make_pipeline(StandardScaler(),
    ...                     LinearSVC(dual="auto", random_state=0, tol=1e-5))
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearsvc', LinearSVC(dual='auto', random_state=0, tol=1e-05))])

    >>> print(clf.named_steps['linearsvc'].coef_)
    [[0.141...   0.526... 0.679... 0.493...]]

    >>> print(clf.named_steps['linearsvc'].intercept_)
    [0.1693...]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """
    _parameter_constraints: dict = {'penalty': [StrOptions({'l1', 'l2'})], 'loss': [StrOptions({'hinge', 'squared_hinge'})], 'dual': ['boolean', StrOptions({'auto'}), Hidden(StrOptions({'warn'}))], 'tol': [Interval(Real, 0.0, None, closed='neither')], 'C': [Interval(Real, 0.0, None, closed='neither')], 'multi_class': [StrOptions({'ovr', 'crammer_singer'})], 'fit_intercept': ['boolean'], 'intercept_scaling': [Interval(Real, 0, None, closed='neither')], 'class_weight': [None, dict, StrOptions({'balanced'})], 'verbose': ['verbose'], 'random_state': ['random_state'], 'max_iter': [Interval(Integral, 0, None, closed='left')]}

    def __init__(self, penalty='l2', loss='squared_hinge', *, dual='warn', tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000):
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=False)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        _dual = _validate_dual_parameter(self.dual, self.loss, self.penalty, self.multi_class, X)
        self.coef_, self.intercept_, n_iter_ = _fit_liblinear(X, y, self.C, self.fit_intercept, self.intercept_scaling, self.class_weight, self.penalty, _dual, self.verbose, self.max_iter, self.tol, self.random_state, self.multi_class, self.loss, sample_weight=sample_weight)
        self.n_iter_ = n_iter_.max().item()
        if self.multi_class == 'crammer_singer' and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])
        return self

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}