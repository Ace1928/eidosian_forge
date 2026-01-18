import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
class NuSVC(BaseSVC):
    """Nu-Support Vector Classification.

    Similar to SVC but uses a parameter to control the number of support
    vectors.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    nu : float, default=0.5
        An upper bound on the fraction of margin errors (see :ref:`User Guide
        <nu_svc>`) and a lower bound of the fraction of support vectors.
        Should be in the interval (0, 1].

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,          default='rbf'
        Specifies the kernel type to be used in the algorithm.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to precompute the kernel matrix. For an intuitive
        visualization of different kernel types see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_kernels.py`.

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

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The "balanced" mode uses the values of y to automatically
        adjust weights inversely proportional to class frequencies as
        ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy. The parameter is
        ignored for binary classification.

        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.

        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.

        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

        .. versionadded:: 0.22

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C of each class.
        Computed based on the ``class_weight`` parameter.

    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes -1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes - 1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    fit_status_ : int
        0 if correctly fitted, 1 if the algorithm did not converge.

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

        .. versionadded:: 1.1

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    fit_status_ : int
        0 if correctly fitted, 1 if the algorithm did not converge.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)

    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    See Also
    --------
    SVC : Support Vector Machine for classification using libsvm.

    LinearSVC : Scalable linear Support Vector Machine for classification using
        liblinear.

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import NuSVC
    >>> clf = make_pipeline(StandardScaler(), NuSVC())
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()), ('nusvc', NuSVC())])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    _impl = 'nu_svc'
    _parameter_constraints: dict = {**BaseSVC._parameter_constraints, 'nu': [Interval(Real, 0.0, 1.0, closed='right')]}
    _parameter_constraints.pop('C')

    def __init__(self, *, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        super().__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=0.0, nu=nu, shrinking=shrinking, probability=probability, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)

    def _more_tags(self):
        return {'_xfail_checks': {'check_methods_subset_invariance': 'fails for the decision_function method', 'check_class_weight_classifiers': 'class_weight is ignored.', 'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples', 'check_classifiers_one_label_sample_weights': 'specified nu is infeasible for the fit.'}}