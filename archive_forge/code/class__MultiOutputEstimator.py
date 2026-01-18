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
class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict'])], 'n_jobs': [Integral, None]}

    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_available_if_estimator_has('partial_fit')
    @_fit_context(prefer_skip_nested_validation=False)
    def partial_fit(self, X, y, classes=None, sample_weight=None, **partial_fit_params):
        """Incrementally fit a separate model for each class output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of ndarray of shape (n_outputs,), default=None
            Each array is unique classes for one output in str/int.
            Can be obtained via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where `y`
            is the target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that `y` doesn't need to contain all labels in `classes`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        _raise_for_params(partial_fit_params, self, 'partial_fit')
        first_time = not hasattr(self, 'estimators_')
        y = self._validate_data(X='no_validation', y=y, multi_output=True)
        if y.ndim == 1:
            raise ValueError('y must have at least two dimensions for multi-output regression but has only one.')
        if _routing_enabled():
            if sample_weight is not None:
                partial_fit_params['sample_weight'] = sample_weight
            routed_params = process_routing(self, 'partial_fit', **partial_fit_params)
        else:
            if sample_weight is not None and (not has_fit_parameter(self.estimator, 'sample_weight')):
                raise ValueError('Underlying estimator does not support sample weights.')
            if sample_weight is not None:
                routed_params = Bunch(estimator=Bunch(partial_fit=Bunch(sample_weight=sample_weight)))
            else:
                routed_params = Bunch(estimator=Bunch(partial_fit=Bunch()))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_partial_fit_estimator)(self.estimators_[i] if not first_time else self.estimator, X, y[:, i], classes[i] if classes is not None else None, partial_fit_params=routed_params.estimator.partial_fit, first_time=first_time) for i in range(y.shape[1])))
        if first_time and hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if first_time and hasattr(self.estimators_[0], 'feature_names_in_'):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        return self

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        if not hasattr(self.estimator, 'fit'):
            raise ValueError('The base estimator should implement a fit method')
        y = self._validate_data(X='no_validation', y=y, multi_output=True)
        if is_classifier(self):
            check_classification_targets(y)
        if y.ndim == 1:
            raise ValueError('y must have at least two dimensions for multi-output regression but has only one.')
        if _routing_enabled():
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            routed_params = process_routing(self, 'fit', **fit_params)
        else:
            if sample_weight is not None and (not has_fit_parameter(self.estimator, 'sample_weight')):
                raise ValueError('Underlying estimator does not support sample weights.')
            fit_params_validated = _check_method_params(X, params=fit_params)
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            if sample_weight is not None:
                routed_params.estimator.fit['sample_weight'] = sample_weight
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_estimator)(self.estimator, X, y[:, i], **routed_params.estimator.fit) for i in range(y.shape[1])))
        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], 'feature_names_in_'):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        return self

    def predict(self, X):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], 'predict'):
            raise ValueError('The base estimator should implement a predict method')
        y = Parallel(n_jobs=self.n_jobs)((delayed(e.predict)(X) for e in self.estimators_))
        return np.asarray(y).T

    def _more_tags(self):
        return {'multioutput_only': True}

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(estimator=self.estimator, method_mapping=MethodMapping().add(callee='partial_fit', caller='partial_fit').add(callee='fit', caller='fit'))
        return router