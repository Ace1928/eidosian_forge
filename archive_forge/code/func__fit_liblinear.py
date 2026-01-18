import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
def _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight, penalty, dual, verbose, max_iter, tol, random_state=None, multi_class='ovr', loss='logistic_regression', epsilon=0.1, sample_weight=None):
    """Used by Logistic Regression (and CV) and LinearSVC/LinearSVR.

    Preprocessing is done in this function before supplying it to liblinear.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X

    C : float
        Inverse of cross-validation parameter. The lower the C, the higher
        the penalization.

    fit_intercept : bool
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: ``[x_1, ..., x_n, 1]``, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float
        Liblinear internally penalizes the intercept, treating it like any
        other term in the feature vector. To reduce the impact of the
        regularization on the intercept, the `intercept_scaling` parameter can
        be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    penalty : {'l1', 'l2'}
        The norm of the penalty used in regularization.

    dual : bool
        Dual or primal formulation,

    verbose : int
        Set verbose to any positive number for verbosity.

    max_iter : int
        Number of iterations.

    tol : float
        Stopping condition.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.

    loss : {'logistic_regression', 'hinge', 'squared_hinge',             'epsilon_insensitive', 'squared_epsilon_insensitive},             default='logistic_regression'
        The loss function used to fit the model.

    epsilon : float, default=0.1
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set epsilon=0.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights assigned to each sample.

    Returns
    -------
    coef_ : ndarray of shape (n_features, n_features + 1)
        The coefficient vector got by minimizing the objective function.

    intercept_ : float
        The intercept term added to the vector.

    n_iter_ : array of int
        Number of iterations run across for each class.
    """
    if loss not in ['epsilon_insensitive', 'squared_epsilon_insensitive']:
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        classes_ = enc.classes_
        if len(classes_) < 2:
            raise ValueError('This solver needs samples of at least 2 classes in the data, but the data contains only one class: %r' % classes_[0])
        class_weight_ = compute_class_weight(class_weight, classes=classes_, y=y)
    else:
        class_weight_ = np.empty(0, dtype=np.float64)
        y_ind = y
    liblinear.set_verbosity_wrap(verbose)
    rnd = check_random_state(random_state)
    if verbose:
        print('[LibLinear]', end='')
    bias = -1.0
    if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError('Intercept scaling is %r but needs to be greater than 0. To disable fitting an intercept, set fit_intercept=False.' % intercept_scaling)
        else:
            bias = intercept_scaling
    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)
    if sp.issparse(X):
        _check_large_sparse(X)
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    y_ind = np.require(y_ind, requirements='W')
    sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(X, y_ind, sp.issparse(X), solver_type, tol, bias, C, class_weight_, max_iter, rnd.randint(np.iinfo('i').max), epsilon, sample_weight)
    n_iter_max = max(n_iter_)
    if n_iter_max >= max_iter:
        warnings.warn('Liblinear failed to converge, increase the number of iterations.', ConvergenceWarning)
    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.0
    return (coef_, intercept_, n_iter_)