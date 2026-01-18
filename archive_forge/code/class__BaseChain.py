from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .base import (
from .model_selection import cross_val_predict
from .utils import Bunch, _print_elapsed_time, check_random_state
from .utils._param_validation import HasMethods, StrOptions
from .utils.metadata_routing import (
from .utils.metaestimators import available_if
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, check_is_fitted, has_fit_parameter
class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {'base_estimator': [HasMethods(['fit', 'predict'])], 'order': ['array-like', StrOptions({'random'}), None], 'cv': ['cv_object', StrOptions({'prefit'})], 'random_state': ['random_state'], 'verbose': ['boolean']}

    def __init__(self, base_estimator, *, order=None, cv=None, random_state=None, verbose=False):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _log_message(self, *, estimator_idx, n_estimators, processing_msg):
        if not self.verbose:
            return None
        return f'({estimator_idx} of {n_estimators}) {processing_msg}'

    @abstractmethod
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True)
        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == 'random':
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError('invalid order')
        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format='lil')
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))
        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format='lil')
        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))
        del Y_pred_chain
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', **fit_params)
        else:
            routed_params = Bunch(estimator=Bunch(fit=fit_params))
        for chain_idx, estimator in enumerate(self.estimators_):
            message = self._log_message(estimator_idx=chain_idx + 1, n_estimators=len(self.estimators_), processing_msg=f'Processing order {self.order_[chain_idx]}')
            y = Y[:, self.order_[chain_idx]]
            with _print_elapsed_time('Chain', message):
                estimator.fit(X_aug[:, :X.shape[1] + chain_idx], y, **routed_params.estimator.fit)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(self.base_estimator, X_aug[:, :col_idx], y=y, cv=self.cv)
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result
        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]
        return Y_pred