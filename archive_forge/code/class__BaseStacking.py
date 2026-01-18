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
class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""
    _parameter_constraints: dict = {'estimators': [list], 'final_estimator': [None, HasMethods('fit')], 'cv': ['cv_object', StrOptions({'prefit'})], 'n_jobs': [None, Integral], 'passthrough': ['boolean'], 'verbose': ['verbose']}

    @abstractmethod
    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, verbose=0, passthrough=False):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _clone_final_estimator(self, default):
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            self.final_estimator_ = clone(default)

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)

        When `y` type is `"multilabel-indicator"`` and the method used is
        `predict_proba`, `preds` can be either a `ndarray` of shape
        `(n_samples, n_class)` or for some estimators a list of `ndarray`.
        This function will drop one of the probability column in this situation as well.
        """
        X_meta = []
        for est_idx, preds in enumerate(predictions):
            if isinstance(preds, list):
                for pred in preds:
                    X_meta.append(pred[:, 1:])
            elif preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
            elif self.stack_method_[est_idx] == 'predict_proba' and len(self.classes_) == 2:
                X_meta.append(preds[:, 1:])
            else:
                X_meta.append(preds)
        self._n_feature_outs = [pred.shape[1] for pred in X_meta]
        if self.passthrough:
            X_meta.append(X)
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)
        return np.hstack(X_meta)

    @staticmethod
    def _method_name(name, estimator, method):
        if estimator == 'drop':
            return None
        if method == 'auto':
            method = ['predict_proba', 'decision_function', 'predict']
        try:
            method_name = _check_response_method(estimator, method).__name__
        except AttributeError as e:
            raise ValueError(f'Underlying estimator {name} does not implement the method {method}.') from e
        return method_name

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()
        stack_method = [self.stack_method] * len(all_estimators)
        if self.cv == 'prefit':
            self.estimators_ = []
            for estimator in all_estimators:
                if estimator != 'drop':
                    check_is_fitted(estimator)
                    self.estimators_.append(estimator)
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_single_estimator)(clone(est), X, y, sample_weight) for est in all_estimators if est != 'drop'))
        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != 'drop':
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, 'feature_names_in_'):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = 'drop'
        self.stack_method_ = [self._method_name(name, est, meth) for name, est, meth in zip(names, all_estimators, stack_method)]
        if self.cv == 'prefit':
            predictions = [getattr(estimator, predict_method)(X) for estimator, predict_method in zip(all_estimators, self.stack_method_) if estimator != 'drop']
        else:
            cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
            if hasattr(cv, 'random_state') and cv.random_state is None:
                cv.random_state = np.random.RandomState()
            fit_params = {'sample_weight': sample_weight} if sample_weight is not None else None
            predictions = Parallel(n_jobs=self.n_jobs)((delayed(cross_val_predict)(clone(est), X, y, cv=deepcopy(cv), method=meth, n_jobs=self.n_jobs, params=fit_params, verbose=self.verbose) for est, meth in zip(all_estimators, self.stack_method_) if est != 'drop'))
        self.stack_method_ = [meth for meth, est in zip(self.stack_method_, all_estimators) if est != 'drop']
        X_meta = self._concatenate_predictions(X, predictions)
        _fit_single_estimator(self.final_estimator_, X_meta, y, sample_weight=sample_weight)
        return self

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(f'{self.__class__.__name__} object has no attribute n_features_in_') from nfe
        return self.estimators_[0].n_features_in_

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        predictions = [getattr(est, meth)(X) for est, meth in zip(self.estimators_, self.stack_method_) if est != 'drop']
        return self._concatenate_predictions(X, predictions)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. The input feature names are only used when `passthrough` is
            `True`.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_ - 1)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

            If `passthrough` is `False`, then only the names of `estimators` are used
            to generate the output feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, 'n_features_in_')
        input_features = _check_feature_names_in(self, input_features, generate_names=self.passthrough)
        class_name = self.__class__.__name__.lower()
        non_dropped_estimators = (name for name, est in self.estimators if est != 'drop')
        meta_names = []
        for est, n_features_out in zip(non_dropped_estimators, self._n_feature_outs):
            if n_features_out == 1:
                meta_names.append(f'{class_name}_{est}')
            else:
                meta_names.extend((f'{class_name}_{est}{i}' for i in range(n_features_out)))
        if self.passthrough:
            return np.concatenate((meta_names, input_features))
        return np.asarray(meta_names, dtype=object)

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
        check_is_fitted(self)
        return self.final_estimator_.predict(self.transform(X), **predict_params)

    def _sk_visual_block_with_final_estimator(self, final_estimator):
        names, estimators = zip(*self.estimators)
        parallel = _VisualBlock('parallel', estimators, names=names, dash_wrapped=False)
        final_block = _VisualBlock('parallel', [final_estimator], names=['final_estimator'], dash_wrapped=False)
        return _VisualBlock('serial', (parallel, final_block), dash_wrapped=False)