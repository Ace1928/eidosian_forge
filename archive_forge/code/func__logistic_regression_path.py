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
def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True, max_iter=100, tol=0.0001, verbose=0, solver='lbfgs', coef=None, class_weight=None, dual=False, penalty='l2', intercept_scaling=1.0, multi_class='auto', random_state=None, check_input=True, max_squared_sum=None, sample_weight=None, l1_ratio=None, n_threads=1):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or array-like of shape (n_cs,), default=10
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'},             default='lbfgs'
        Numerical solver to use.

    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default=1.
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

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
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
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    n_threads : int, default=1
       Number of OpenMP threads to use.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)
    solver = _check_solver(solver, penalty, dual)
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=solver not in ['liblinear', 'sag', 'saga'])
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    n_samples, n_features = X.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)
    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != 'multinomial':
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')
        pos_class = classes[1]
    if sample_weight is not None or class_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)
    le = LabelEncoder()
    if isinstance(class_weight, dict) or (multi_class == 'multinomial' and class_weight is not None):
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask = y == pos_class
        y_bin = np.ones(y.shape, dtype=X.dtype)
        if solver in ['lbfgs', 'newton-cg', 'newton-cholesky']:
            mask_classes = np.array([0, 1])
            y_bin[~mask] = 0.0
        else:
            mask_classes = np.array([-1, 1])
            y_bin[~mask] = -1.0
        if class_weight == 'balanced':
            class_weight_ = compute_class_weight(class_weight, classes=mask_classes, y=y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]
    else:
        if solver in ['sag', 'saga', 'lbfgs', 'newton-cg']:
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)
        else:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        w0 = np.zeros((classes.size, n_features + int(fit_intercept)), order='F', dtype=X.dtype)
    if solver in ['lbfgs', 'newton-cg', 'newton-cholesky']:
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
    if coef is not None:
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError('Initialization coef is of shape %d, expected shape %d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1
            if coef.shape[0] != n_classes or coef.shape[1] not in (n_features, n_features + 1):
                raise ValueError('Initialization coef is of shape (%d, %d), expected shape (%d, %d) or (%d, %d)' % (coef.shape[0], coef.shape[1], classes.size, n_features, classes.size, n_features + 1))
            if n_classes == 1:
                w0[0, :coef.shape[1]] = -coef
                w0[1, :coef.shape[1]] = coef
            else:
                w0[:, :coef.shape[1]] = coef
    if multi_class == 'multinomial':
        if solver in ['lbfgs', 'newton-cg']:
            w0 = w0.ravel(order='F')
            loss = LinearModelLoss(base_loss=HalfMultinomialLoss(n_classes=classes.size), fit_intercept=fit_intercept)
        target = Y_multi
        if solver == 'lbfgs':
            func = loss.loss_gradient
        elif solver == 'newton-cg':
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            loss = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept)
            func = loss.loss_gradient
        elif solver == 'newton-cg':
            loss = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept)
            func = loss.loss
            grad = loss.gradient
            hess = loss.gradient_hessian_product
        elif solver == 'newton-cholesky':
            loss = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept)
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}
    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            l2_reg_strength = 1.0 / (C * sw_sum)
            iprint = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
            opt_res = optimize.minimize(func, w0, method='L-BFGS-B', jac=True, args=(X, target, sample_weight, l2_reg_strength, n_threads), options={'maxiter': max_iter, 'maxls': 50, 'iprint': iprint, 'gtol': tol, 'ftol': 64 * np.finfo(float).eps})
            n_iter_i = _check_optimize_result(solver, opt_res, max_iter, extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
            w0, loss = (opt_res.x, opt_res.fun)
        elif solver == 'newton-cg':
            l2_reg_strength = 1.0 / (C * sw_sum)
            args = (X, target, sample_weight, l2_reg_strength, n_threads)
            w0, n_iter_i = _newton_cg(hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol)
        elif solver == 'newton-cholesky':
            l2_reg_strength = 1.0 / (C * sw_sum)
            sol = NewtonCholeskySolver(coef=w0, linear_loss=loss, l2_reg_strength=l2_reg_strength, tol=tol, max_iter=max_iter, n_threads=n_threads, verbose=verbose)
            w0 = sol.solve(X=X, y=target, sample_weight=sample_weight)
            n_iter_i = sol.iteration
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i = _fit_liblinear(X, target, C, fit_intercept, intercept_scaling, None, penalty, dual, verbose, max_iter, tol, random_state, sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()
            n_iter_i = n_iter_i.item()
        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(X.dtype, copy=False)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.0
                beta = 1.0 / C
            elif penalty == 'l2':
                alpha = 1.0 / C
                beta = 0.0
            else:
                alpha = 1.0 / C * (1 - l1_ratio)
                beta = 1.0 / C * l1_ratio
            w0, n_iter_i, warm_start_sag = sag_solver(X, target, sample_weight, loss, alpha, beta, max_iter, tol, verbose, random_state, False, max_squared_sum, warm_start_sag, is_saga=solver == 'saga')
        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', 'newton-cg', 'sag'}, got '%s' instead" % solver)
        if multi_class == 'multinomial':
            n_classes = max(2, classes.size)
            if solver in ['lbfgs', 'newton-cg']:
                multi_w0 = np.reshape(w0, (n_classes, -1), order='F')
            else:
                multi_w0 = w0
            if n_classes == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0.copy())
        else:
            coefs.append(w0.copy())
        n_iter[i] = n_iter_i
    return (np.array(coefs), np.array(Cs), n_iter)