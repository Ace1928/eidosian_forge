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
def _log_reg_scoring_path(X, y, train, test, *, pos_class, Cs, scoring, fit_intercept, max_iter, tol, class_weight, verbose, solver, penalty, dual, intercept_scaling, multi_class, random_state, max_squared_sum, sample_weight, l1_ratio, score_params):
    """Computes scores across logistic_regression_path

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target labels.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    pos_class : int
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or list of floats
        Each of the values in Cs describes the inverse of
        regularization strength. If Cs is as an int, then a grid of Cs
        values are chosen in a logarithmic scale between 1e-4 and 1e4.

    scoring : callable
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`.

    fit_intercept : bool
        If False, then the bias term is set to zero. Else the last
        term of each coef_ gives us the intercept.

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Tolerance for stopping criteria.

    class_weight : dict or 'balanced'
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}
        Decides which solver to use.

    penalty : {'l1', 'l2', 'elasticnet'}
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    intercept_scaling : float
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'auto', 'ovr', 'multinomial'}
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.

    random_state : int, RandomState instance
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    max_squared_sum : float
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,)
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    score_params : dict
        Parameters to pass to the `score` method of the underlying scorer.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    scores : ndarray of shape (n_cs,)
        Scores obtained for each Cs.

    n_iter : ndarray of shape(n_cs,)
        Actual number of iteration for each Cs.
    """
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        sample_weight = sample_weight[train]
    coefs, Cs, n_iter = _logistic_regression_path(X_train, y_train, Cs=Cs, l1_ratio=l1_ratio, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter, class_weight=class_weight, pos_class=pos_class, multi_class=multi_class, tol=tol, verbose=verbose, dual=dual, penalty=penalty, intercept_scaling=intercept_scaling, random_state=random_state, check_input=False, max_squared_sum=max_squared_sum, sample_weight=sample_weight)
    log_reg = LogisticRegression(solver=solver, multi_class=multi_class)
    if multi_class == 'ovr':
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == 'multinomial':
        log_reg.classes_ = np.unique(y_train)
    else:
        raise ValueError('multi_class should be either multinomial or ovr, got %d' % multi_class)
    if pos_class is not None:
        mask = y_test == pos_class
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.0
    scores = list()
    scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == 'ovr':
            w = w[np.newaxis, :]
        if fit_intercept:
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            log_reg.coef_ = w
            log_reg.intercept_ = 0.0
        if scoring is None:
            scores.append(log_reg.score(X_test, y_test))
        else:
            score_params = score_params or {}
            score_params = _check_method_params(X=X, params=score_params, indices=test)
            scores.append(scoring(log_reg, X_test, y_test, **score_params))
    return (coefs, Cs, np.array(scores), n_iter)