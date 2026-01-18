from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numbers import Integral
import numpy as np
import scipy.sparse as sparse
from ..base import (
from ..exceptions import NotFittedError
from ..linear_model import LogisticRegression, RidgeCV
from ..model_selection import check_cv, cross_val_predict
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import HasMethods, StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
class StackingClassifier(_RoutingNotSupportedMixin, ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

        The type of estimator is generally expected to be a classifier.
        However, one can pass a regressor for some use case (e.g. ordinal
        regression).

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.

    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits,
        * `"prefit"` to assume the `estimators` are prefit. In this case, the
          estimators will not be refitted.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that all `estimators` have
        been fitted already. The `final_estimator_` is trained on the `estimators`
        predictions on the full training set and are **not** cross validated
        predictions. Please note that if the models have been trained on the same
        data to train the stacking model, there is a very high risk of overfitting.

        .. versionadded:: 1.1
            The 'prefit' option was added in 1.1

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'},             default='auto'
        Methods called for each base estimator. It can be:

        * if 'auto', it will try to invoke, for each estimator,
          `'predict_proba'`, `'decision_function'` or `'predict'` in that
          order.
        * otherwise, one of `'predict_proba'`, `'decision_function'` or
          `'predict'`. If the method is not implemented by the estimator, it
          will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel all `estimators` `fit`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See Glossary for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray if `y`         is of type `"multilabel-indicator"`.
        Class labels.

    estimators_ : list of estimators
        The elements of the `estimators` parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    final_estimator_ : estimator
        The classifier which predicts given the output of `estimators_`.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingRegressor : Stack of estimators with a final regressor.

    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    The first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.

    In some cases (e.g. ordinal regression), one can pass regressors as the
    first layer of the :class:`StackingClassifier`. However, note that `y` will
    be internally encoded in a numerically increasing order or lexicographic
    order. If this ordering is not adequate, one should manually numerically
    encode the classes in the desired order.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(dual="auto", random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...
    """
    _parameter_constraints: dict = {**_BaseStacking._parameter_constraints, 'stack_method': [StrOptions({'auto', 'predict_proba', 'decision_function', 'predict'})]}

    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0):
        super().__init__(estimators=estimators, final_estimator=final_estimator, cv=cv, stack_method=stack_method, n_jobs=n_jobs, passthrough=passthrough, verbose=verbose)

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError("'final_estimator' parameter should be a classifier. Got {}".format(self.final_estimator_))

    def _validate_estimators(self):
        """Overload the method of `_BaseHeterogeneousEnsemble` to be more
        lenient towards the type of `estimators`.

        Regressors can be accepted for some cases such as ordinal regression.
        """
        if len(self.estimators) == 0:
            raise ValueError("Invalid 'estimators' attribute, 'estimators' should be a non-empty list of (string, estimator) tuples.")
        names, estimators = zip(*self.estimators)
        self._validate_names(names)
        has_estimator = any((est != 'drop' for est in estimators))
        if not has_estimator:
            raise ValueError('All estimators are dropped. At least one is required to be an estimator.')
        return (names, estimators)

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values. Note that `y` will be internally encoded in
            numerically increasing order or lexicographic order. If the order
            matter (e.g. for ordinal regression), one should numerically encode
            the target `y` before calling :term:`fit`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        check_classification_targets(y)
        if type_of_target(y) == 'multilabel-indicator':
            self._label_encoder = [LabelEncoder().fit(yk) for yk in y.T]
            self.classes_ = [le.classes_ for le in self._label_encoder]
            y_encoded = np.array([self._label_encoder[target_idx].transform(target) for target_idx, target in enumerate(y.T)]).T
        else:
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            y_encoded = self._label_encoder.transform(y)
        return super().fit(X, y_encoded, sample_weight)

    @available_if(_estimator_has('predict'))
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        y_pred = super().predict(X, **predict_params)
        if isinstance(self._label_encoder, list):
            y_pred = np.array([self._label_encoder[target_idx].inverse_transform(target) for target_idx, target in enumerate(y_pred.T)]).T
        else:
            y_pred = self._label_encoder.inverse_transform(y_pred)
        return y_pred

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X):
        """Predict class probabilities for `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or             list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        y_pred = self.final_estimator_.predict_proba(self.transform(X))
        if isinstance(self._label_encoder, list):
            y_pred = np.array([preds[:, 0] for preds in y_pred]).T
        return y_pred

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X):
        """Decision function for samples in `X` using the final estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes),             or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        check_is_fitted(self)
        return self.final_estimator_.decision_function(self.transform(X))

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or                 (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X)

    def _sk_visual_block_(self):
        if self.final_estimator is None:
            final_estimator = LogisticRegression()
        else:
            final_estimator = self.final_estimator
        return super()._sk_visual_block_with_final_estimator(final_estimator)