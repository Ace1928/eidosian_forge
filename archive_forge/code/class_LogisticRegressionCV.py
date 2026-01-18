import numbers
import warnings
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import optimize
from sklearn.metrics import get_scorer_names
from .._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
from ..base import _fit_context
from ..metrics import get_scorer
from ..model_selection import check_cv
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..svm._base import _fit_liblinear
from ..utils import (
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import row_norms, softmax
from ..utils.metadata_routing import (
from ..utils.multiclass import check_classification_targets
from ..utils.optimize import _check_optimize_result, _newton_cg
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEstimator, LinearClassifierMixin, SparseCoefMixin
from ._glm.glm import NewtonCholeskySolver
from ._linear_loss import LinearModelLoss
from ._sag import sag_solver
class LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator):
    """Logistic Regression CV (aka logit, MaxEnt) classifier.

    See glossary entry for :term:`cross-validation estimator`.

    This class implements logistic regression using liblinear, newton-cg, sag
    of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
    regularization with primal formulation. The liblinear solver supports both
    L1 and L2 regularization, with a dual formulation only for the L2 penalty.
    Elastic-Net penalty is only supported by the saga solver.

    For the grid of `Cs` values and `l1_ratios` values, the best hyperparameter
    is selected by the cross-validator
    :class:`~sklearn.model_selection.StratifiedKFold`, but it can be changed
    using the :term:`cv` parameter. The 'newton-cg', 'sag', 'saga' and 'lbfgs'
    solvers can warm-start the coefficients (see :term:`Glossary<warm_start>`).

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    cv : int or cross-validation generator, default=None
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    dual : bool, default=False
        Dual (constrained) or primal (regularized, see also
        :ref:`this equation <regularized-logistic-loss>`) formulation. Dual formulation
        is only implemented for l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Specify the norm of the penalty:

        - `'l2'`: add a L2 penalty term (used by default);
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is 'accuracy'.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'},             default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' might be slower in :class:`LogisticRegressionCV`
              because it does not handle warm-starting. 'liblinear' is
              limited to one-versus-rest schemes.
            - 'newton-cholesky' is a good choice for `n_samples` >> `n_features`,
              especially with one-hot encoded categorical features with rare
              categories. Note that it is limited to binary classification and the
              one-versus-rest reduction for multiclass classification. Be aware that
              the memory usage of this solver has a quadratic dependency on
              `n_features` because it explicitly computes the Hessian matrix.

        .. warning::
           The choice of the algorithm depends on the penalty chosen.
           Supported penalties by solver:

           - 'lbfgs'           -   ['l2']
           - 'liblinear'       -   ['l1', 'l2']
           - 'newton-cg'       -   ['l2']
           - 'newton-cholesky' -   ['l2']
           - 'sag'             -   ['l2']
           - 'saga'            -   ['elasticnet', 'l1', 'l2']

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionadded:: 1.2
           newton-cholesky solver.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    max_iter : int, default=100
        Maximum number of iterations of the optimization algorithm.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           class_weight == 'balanced'

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
        positive number for verbosity.

    refit : bool, default=True
        If set to True, the scores are averaged across all folds, and the
        coefs and the C that corresponds to the best score is taken, and a
        final refit is done using these parameters.
        Otherwise the coefs, intercepts and C that correspond to the
        best scores across folds are averaged.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'auto, 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when `solver='sag'`, 'saga' or 'liblinear' to shuffle the data.
        Note that this only applies to the solver and not the cross-validation
        generator. See :term:`Glossary <random_state>` for details.

    l1_ratios : list of float, default=None
        The list of Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``.
        Only used if ``penalty='elasticnet'``. A value of 0 is equivalent to
        using ``penalty='l2'``, while 1 is equivalent to using
        ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a combination
        of L1 and L2.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem
        is binary.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.

    Cs_ : ndarray of shape (n_cs)
        Array of C i.e. inverse of regularization parameter values used
        for cross-validation.

    l1_ratios_ : ndarray of shape (n_l1_ratios)
        Array of l1_ratios used for cross-validation. If no l1_ratio is used
        (i.e. penalty is not 'elasticnet'), this is set to ``[None]``

    coefs_paths_ : ndarray of shape (n_folds, n_cs, n_features) or                    (n_folds, n_cs, n_features + 1)
        dict with classes as the keys, and the path of coefficients obtained
        during cross-validating across each fold and then across each Cs
        after doing an OvR for the corresponding class as values.
        If the 'multi_class' option is set to 'multinomial', then
        the coefs_paths are the coefficients corresponding to each class.
        Each dict value has shape ``(n_folds, n_cs, n_features)`` or
        ``(n_folds, n_cs, n_features + 1)`` depending on whether the
        intercept is fit or not. If ``penalty='elasticnet'``, the shape is
        ``(n_folds, n_cs, n_l1_ratios_, n_features)`` or
        ``(n_folds, n_cs, n_l1_ratios_, n_features + 1)``.

    scores_ : dict
        dict with classes as the keys, and the values as the
        grid of scores obtained during cross-validating each fold, after doing
        an OvR for the corresponding class. If the 'multi_class' option
        given is 'multinomial' then the same scores are repeated across
        all classes, since this is the multinomial class. Each dict value
        has shape ``(n_folds, n_cs)`` or ``(n_folds, n_cs, n_l1_ratios)`` if
        ``penalty='elasticnet'``.

    C_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of C that maps to the best scores across every class. If refit is
        set to False, then for each class, the best C is the average of the
        C's that correspond to the best scores for each fold.
        `C_` is of shape(n_classes,) when the problem is binary.

    l1_ratio_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of l1_ratio that maps to the best scores across every class. If
        refit is set to False, then for each class, the best l1_ratio is the
        average of the l1_ratio's that correspond to the best scores for each
        fold.  `l1_ratio_` is of shape(n_classes,) when the problem is binary.

    n_iter_ : ndarray of shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
        Actual number of iterations for all classes, folds and Cs.
        In the binary or multinomial cases, the first dimension is equal to 1.
        If ``penalty='elasticnet'``, the shape is ``(n_classes, n_folds,
        n_cs, n_l1_ratios)`` or ``(1, n_folds, n_cs, n_l1_ratios)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    LogisticRegression : Logistic regression without tuning the
        hyperparameter `C`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegressionCV
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :]).shape
    (2, 3)
    >>> clf.score(X, y)
    0.98...
    """
    _parameter_constraints: dict = {**LogisticRegression._parameter_constraints}
    for param in ['C', 'warm_start', 'l1_ratio']:
        _parameter_constraints.pop(param)
    _parameter_constraints.update({'Cs': [Interval(Integral, 1, None, closed='left'), 'array-like'], 'cv': ['cv_object'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'l1_ratios': ['array-like', None], 'refit': ['boolean'], 'penalty': [StrOptions({'l1', 'l2', 'elasticnet'})]})

    def __init__(self, *, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None):
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
        self.l1_ratios = l1_ratios

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, **params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        **params : dict
            Parameters to pass to the underlying splitter and scorer.

            .. versionadded:: 1.4

        Returns
        -------
        self : object
            Fitted LogisticRegressionCV estimator.
        """
        _raise_for_params(params, self, 'fit')
        solver = _check_solver(self.solver, self.penalty, self.dual)
        if self.penalty == 'elasticnet':
            if self.l1_ratios is None or len(self.l1_ratios) == 0 or any((not isinstance(l1_ratio, numbers.Number) or l1_ratio < 0 or l1_ratio > 1 for l1_ratio in self.l1_ratios)):
                raise ValueError('l1_ratios must be a list of numbers between 0 and 1; got (l1_ratios=%r)' % self.l1_ratios)
            l1_ratios_ = self.l1_ratios
        else:
            if self.l1_ratios is not None:
                warnings.warn("l1_ratios parameter is only used when penalty is 'elasticnet'. Got (penalty={})".format(self.penalty))
            l1_ratios_ = [None]
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=solver not in ['liblinear', 'sag', 'saga'])
        check_classification_targets(y)
        class_weight = self.class_weight
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        if isinstance(class_weight, dict):
            class_weight = {label_encoder.transform([cls])[0]: v for cls, v in class_weight.items()}
        classes = self.classes_ = label_encoder.classes_
        encoded_labels = label_encoder.transform(label_encoder.classes_)
        multi_class = _check_multi_class(self.multi_class, solver, len(classes))
        if solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', sample_weight=sample_weight, **params)
        else:
            routed_params = Bunch()
            routed_params.splitter = Bunch(split={})
            routed_params.scorer = Bunch(score=params)
            if sample_weight is not None:
                routed_params.scorer.score['sample_weight'] = sample_weight
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y, **routed_params.splitter.split))
        n_classes = len(encoded_labels)
        if n_classes < 2:
            raise ValueError('This solver needs samples of at least 2 classes in the data, but the data contains only one class: %r' % classes[0])
        if n_classes == 2:
            n_classes = 1
            encoded_labels = encoded_labels[1:]
            classes = classes[1:]
        if multi_class == 'multinomial':
            iter_encoded_labels = iter_classes = [None]
        else:
            iter_encoded_labels = encoded_labels
            iter_classes = classes
        if class_weight == 'balanced':
            class_weight = compute_class_weight(class_weight, classes=np.arange(len(self.classes_)), y=y)
            class_weight = dict(enumerate(class_weight))
        path_func = delayed(_log_reg_scoring_path)
        if self.solver in ['sag', 'saga']:
            prefer = 'threads'
        else:
            prefer = 'processes'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer)((path_func(X, y, train, test, pos_class=label, Cs=self.Cs, fit_intercept=self.fit_intercept, penalty=self.penalty, dual=self.dual, solver=solver, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose, class_weight=class_weight, scoring=self.scoring, multi_class=multi_class, intercept_scaling=self.intercept_scaling, random_state=self.random_state, max_squared_sum=max_squared_sum, sample_weight=sample_weight, l1_ratio=l1_ratio, score_params=routed_params.scorer.score) for label in iter_encoded_labels for train, test in folds for l1_ratio in l1_ratios_))
        coefs_paths, Cs, scores, n_iter_ = zip(*fold_coefs_)
        self.Cs_ = Cs[0]
        if multi_class == 'multinomial':
            coefs_paths = np.reshape(coefs_paths, (len(folds), len(l1_ratios_) * len(self.Cs_), n_classes, -1))
            coefs_paths = np.swapaxes(coefs_paths, 0, 1)
            coefs_paths = np.swapaxes(coefs_paths, 0, 2)
            self.n_iter_ = np.reshape(n_iter_, (1, len(folds), len(self.Cs_) * len(l1_ratios_)))
            scores = np.tile(scores, (n_classes, 1, 1))
        else:
            coefs_paths = np.reshape(coefs_paths, (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_), -1))
            self.n_iter_ = np.reshape(n_iter_, (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_)))
        scores = np.reshape(scores, (n_classes, len(folds), -1))
        self.scores_ = dict(zip(classes, scores))
        self.coefs_paths_ = dict(zip(classes, coefs_paths))
        self.C_ = list()
        self.l1_ratio_ = list()
        self.coef_ = np.empty((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        for index, (cls, encoded_label) in enumerate(zip(iter_classes, iter_encoded_labels)):
            if multi_class == 'ovr':
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]
            else:
                scores = scores[0]
            if self.refit:
                best_index = scores.sum(axis=0).argmax()
                best_index_C = best_index % len(self.Cs_)
                C_ = self.Cs_[best_index_C]
                self.C_.append(C_)
                best_index_l1 = best_index // len(self.Cs_)
                l1_ratio_ = l1_ratios_[best_index_l1]
                self.l1_ratio_.append(l1_ratio_)
                if multi_class == 'multinomial':
                    coef_init = np.mean(coefs_paths[:, :, best_index, :], axis=1)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)
                w, _, _ = _logistic_regression_path(X, y, pos_class=encoded_label, Cs=[C_], solver=solver, fit_intercept=self.fit_intercept, coef=coef_init, max_iter=self.max_iter, tol=self.tol, penalty=self.penalty, class_weight=class_weight, multi_class=multi_class, verbose=max(0, self.verbose - 1), random_state=self.random_state, check_input=False, max_squared_sum=max_squared_sum, sample_weight=sample_weight, l1_ratio=l1_ratio_)
                w = w[0]
            else:
                best_indices = np.argmax(scores, axis=1)
                if multi_class == 'ovr':
                    w = np.mean([coefs_paths[i, best_indices[i], :] for i in range(len(folds))], axis=0)
                else:
                    w = np.mean([coefs_paths[:, i, best_indices[i], :] for i in range(len(folds))], axis=0)
                best_indices_C = best_indices % len(self.Cs_)
                self.C_.append(np.mean(self.Cs_[best_indices_C]))
                if self.penalty == 'elasticnet':
                    best_indices_l1 = best_indices // len(self.Cs_)
                    self.l1_ratio_.append(np.mean(l1_ratios_[best_indices_l1]))
                else:
                    self.l1_ratio_.append(None)
            if multi_class == 'multinomial':
                self.C_ = np.tile(self.C_, n_classes)
                self.l1_ratio_ = np.tile(self.l1_ratio_, n_classes)
                self.coef_ = w[:, :X.shape[1]]
                if self.fit_intercept:
                    self.intercept_ = w[:, -1]
            else:
                self.coef_[index] = w[:X.shape[1]]
                if self.fit_intercept:
                    self.intercept_[index] = w[-1]
        self.C_ = np.asarray(self.C_)
        self.l1_ratio_ = np.asarray(self.l1_ratio_)
        self.l1_ratios_ = np.asarray(l1_ratios_)
        if self.l1_ratios is not None:
            for cls, coefs_path in self.coefs_paths_.items():
                self.coefs_paths_[cls] = coefs_path.reshape((len(folds), self.l1_ratios_.size, self.Cs_.size, -1))
                self.coefs_paths_[cls] = np.transpose(self.coefs_paths_[cls], (0, 2, 1, 3))
            for cls, score in self.scores_.items():
                self.scores_[cls] = score.reshape((len(folds), self.l1_ratios_.size, self.Cs_.size))
                self.scores_[cls] = np.transpose(self.scores_[cls], (0, 2, 1))
            self.n_iter_ = self.n_iter_.reshape((-1, len(folds), self.l1_ratios_.size, self.Cs_.size))
            self.n_iter_ = np.transpose(self.n_iter_, (0, 1, 3, 2))
        return self

    def score(self, X, y, sample_weight=None, **score_params):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying scorer.

            .. versionadded:: 1.4

        Returns
        -------
        score : float
            Score of self.predict(X) w.r.t. y.
        """
        _raise_for_params(score_params, self, 'score')
        scoring = self._get_scorer()
        if _routing_enabled():
            routed_params = process_routing(self, 'score', sample_weight=sample_weight, **score_params)
        else:
            routed_params = Bunch()
            routed_params.scorer = Bunch(score={})
            if sample_weight is not None:
                routed_params.scorer.score['sample_weight'] = sample_weight
        return scoring(self, X, y, **routed_params.scorer.score)

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(splitter=self.cv, method_mapping=MethodMapping().add(callee='split', caller='fit')).add(scorer=self._get_scorer(), method_mapping=MethodMapping().add(callee='score', caller='score').add(callee='score', caller='fit'))
        return router

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}

    def _get_scorer(self):
        """Get the scorer based on the scoring method specified.
        The default scoring method is `accuracy`.
        """
        scoring = self.scoring or 'accuracy'
        return get_scorer(scoring)