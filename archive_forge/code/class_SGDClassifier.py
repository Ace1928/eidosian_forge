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
class SGDClassifier(BaseSGDClassifier):
    """Linear classifiers (SVM, logistic regression, etc.) with SGD training.

    This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning via the `partial_fit` method.
    For best results using the default learning rate schedule, the data should
    have zero mean and unit variance.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : {'hinge', 'log_loss', 'modified_huber', 'squared_hinge',        'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',        'squared_epsilon_insensitive'}, default='hinge'
        The loss function to be used.

        - 'hinge' gives a linear SVM.
        - 'log_loss' gives logistic regression, a probabilistic classifier.
        - 'modified_huber' is another smooth loss that brings tolerance to
          outliers as well as probability estimates.
        - 'squared_hinge' is like hinge but is quadratically penalized.
        - 'perceptron' is the linear loss used by the perceptron algorithm.
        - The other losses, 'squared_error', 'huber', 'epsilon_insensitive' and
          'squared_epsilon_insensitive' are designed for regression but can be useful
          in classification as well; see
          :class:`~sklearn.linear_model.SGDRegressor` for a description.

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

    penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'. No penalty is added when set to `None`.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization. Also used to compute the
        learning rate when `learning_rate` is set to 'optimal'.
        Values must be in the range `[0.0, inf)`.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.
        Values must be in the range `[0.0, 1.0]`.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.
        Values must be in the range `[1, inf)`.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, training will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.
        Values must be in the range `[0, inf)`.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.
        Values must be in the range `[0.0, inf)`.

    n_jobs : int, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
        Integer values must be in the range `[0, 2**32 - 1]`.

    learning_rate : str, default='optimal'
        The learning rate schedule:

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where `t0` is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': `eta = eta0`, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          `early_stopping` is `True`, the current learning rate is divided by 5.

            .. versionadded:: 0.20
                Added 'adaptive' option

    eta0 : float, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.
        Values must be in the range `[0.0, inf)`.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        Values must be in the range `(-inf, inf)`.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to `True`, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least tol for n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20
            Added 'early_stopping' option

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.
        Values must be in the range `(0.0, 1.0)`.

        .. versionadded:: 0.20
            Added 'validation_fraction' option

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.
        Integer values must be in the range `[1, max_iter)`.

        .. versionadded:: 0.20
            Added 'n_iter_no_change' option

    class_weight : dict, {class_label: weight} or "balanced", default=None
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit`` will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to `True`, computes the averaged SGD weights across all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.
        Integer values must be in the range `[1, n_samples]`.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else             (n_classes, n_features)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    loss_function_ : concrete ``LossFunction``

        .. deprecated:: 1.4
            Attribute `loss_function_` was deprecated in version 1.4 and will be
            removed in 1.6.

    classes_ : array of shape (n_classes,)

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.svm.LinearSVC : Linear support vector classification.
    LogisticRegression : Logistic regression.
    Perceptron : Inherits from SGDClassifier. ``Perceptron()`` is equivalent to
        ``SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant",
        penalty=None)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> # Always scale the input. The most convenient way is to use a pipeline.
    >>> clf = make_pipeline(StandardScaler(),
    ...                     SGDClassifier(max_iter=1000, tol=1e-3))
    >>> clf.fit(X, Y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('sgdclassifier', SGDClassifier())])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    _parameter_constraints: dict = {**BaseSGDClassifier._parameter_constraints, 'penalty': [StrOptions({'l2', 'l1', 'elasticnet'}), None], 'alpha': [Interval(Real, 0, None, closed='left')], 'l1_ratio': [Interval(Real, 0, 1, closed='both')], 'power_t': [Interval(Real, None, None, closed='neither')], 'epsilon': [Interval(Real, 0, None, closed='left')], 'learning_rate': [StrOptions({'constant', 'optimal', 'invscaling', 'adaptive'}), Hidden(StrOptions({'pa1', 'pa2'}))], 'eta0': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start, average=average)

    def _check_proba(self):
        if self.loss not in ('log_loss', 'modified_huber'):
            raise AttributeError('probability estimates are not available for loss=%r' % self.loss)
        return True

    @available_if(_check_proba)
    def predict_proba(self, X):
        """Probability estimates.

        This method is only available for log loss and modified Huber loss.

        Multiclass probability estimates are derived from binary (one-vs.-rest)
        estimates by simple normalization, as recommended by Zadrozny and
        Elkan.

        Binary probability estimates for loss="modified_huber" are given by
        (clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions
        it is necessary to perform proper probability calibration by wrapping
        the classifier with
        :class:`~sklearn.calibration.CalibratedClassifierCV` instead.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.

        References
        ----------
        Zadrozny and Elkan, "Transforming classifier scores into multiclass
        probability estimates", SIGKDD'02,
        https://dl.acm.org/doi/pdf/10.1145/775047.775151

        The justification for the formula in the loss="modified_huber"
        case is in the appendix B in:
        http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        check_is_fitted(self)
        if self.loss == 'log_loss':
            return self._predict_proba_lr(X)
        elif self.loss == 'modified_huber':
            binary = len(self.classes_) == 2
            scores = self.decision_function(X)
            if binary:
                prob2 = np.ones((scores.shape[0], 2))
                prob = prob2[:, 1]
            else:
                prob = scores
            np.clip(scores, -1, 1, prob)
            prob += 1.0
            prob /= 2.0
            if binary:
                prob2[:, 0] -= prob
                prob = prob2
            else:
                prob_sum = prob.sum(axis=1)
                all_zero = prob_sum == 0
                if np.any(all_zero):
                    prob[all_zero, :] = 1
                    prob_sum[all_zero] = len(self.classes_)
                prob /= prob_sum.reshape((prob.shape[0], -1))
            return prob
        else:
            raise NotImplementedError("predict_(log_)proba only supported when loss='log_loss' or loss='modified_huber' (%r given)" % self.loss)

    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Log of probability estimates.

        This method is only available for log loss and modified Huber loss.

        When loss="modified_huber", probability estimates may be hard zeros
        and ones, so taking the logarithm is not possible.

        See ``predict_proba`` for details.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        T : array-like, shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in
            `self.classes_`.
        """
        return np.log(self.predict_proba(X))

    def _more_tags(self):
        return {'_xfail_checks': {'check_sample_weights_invariance': 'zero sample_weight is not equivalent to removing samples'}, 'preserves_dtype': [np.float64, np.float32]}