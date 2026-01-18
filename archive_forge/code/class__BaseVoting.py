from abc import abstractmethod
from numbers import Integral
import numpy as np
from ..base import (
from ..exceptions import NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimators': [list], 'weights': ['array-like', None], 'n_jobs': [None, Integral], 'verbose': ['verbose']}

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return f'({idx} of {total}) Processing {name}'

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != 'drop']

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X) for est in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(f'Number of `estimators` and weights must be equal; got {len(self.weights)} weights, {len(self.estimators)} estimators')
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_single_estimator)(clone(clf), X, y, sample_weight=sample_weight, message_clsname='Voting', message=self._log_message(names[idx], idx + 1, len(clfs))) for idx, clf in enumerate(clfs) if clf != 'drop'))
        self.named_estimators_ = Bunch()
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == 'drop' else next(est_iter)
            self.named_estimators_[name] = current_est
            if hasattr(current_est, 'feature_names_in_'):
                self.feature_names_in_ = current_est.feature_names_in_
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return class labels or probabilities for each estimator.

        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape                 (n_samples, n_features)
            Input samples.

        y : ndarray of shape (n_samples,), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return super().fit_transform(X, y, **fit_params)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError('{} object has no n_features_in_ attribute.'.format(self.__class__.__name__)) from nfe
        return self.estimators_[0].n_features_in_

    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        return _VisualBlock('parallel', estimators, names=names)