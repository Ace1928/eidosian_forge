import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
class NuSVR(RegressorMixin, BaseLibSVM):
    """Nu Support Vector Regression.

    Similar to NuSVC, for regression, uses a parameter nu to control
    the number of support vectors. However, unlike NuSVC, where nu
    replaces C, here nu replaces the parameter epsilon of epsilon-SVR.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    nu : float, default=0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Should be in the interval (0, 1].  By
        default 0.5 will be taken.

    C : float, default=1.0
        Penalty parameter C of the error term.

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

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

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
        Coefficients of the support vector in the decision function.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (1,)
        Constants in decision function.

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

    n_support_ : ndarray of shape (1,), dtype=int32
        Number of support vectors.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    See Also
    --------
    NuSVC : Support Vector Machine for classification implemented with libsvm
        with a parameter to control the number of support vectors.

    SVR : Epsilon Support Vector Machine for regression implemented with
        libsvm.

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_

    Examples
    --------
    >>> from sklearn.svm import NuSVR
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
    >>> regr.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('nusvr', NuSVR(nu=0.1))])
    """
    _impl = 'nu_svr'
    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints}
    for unused_param in ['class_weight', 'epsilon', 'probability', 'random_state']:
        _parameter_constraints.pop(unused_param)

    def __init__(self, *, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
        super().__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0.0, shrinking=shrinking, probability=False, cache_size=cache_size, class_weight=None, verbose=verbose, max_iter=max_iter, random_state=None)

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}}