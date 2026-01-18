import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(self, boosting_type: str='gbdt', num_leaves: int=31, max_depth: int=-1, learning_rate: float=0.1, n_estimators: int=100, subsample_for_bin: int=200000, objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]]=None, class_weight: Optional[Union[Dict, str]]=None, min_split_gain: float=0.0, min_child_weight: float=0.001, min_child_samples: int=20, subsample: float=1.0, subsample_freq: int=0, colsample_bytree: float=1.0, reg_alpha: float=0.0, reg_lambda: float=0.0, random_state: Optional[Union[int, np.random.RandomState, 'np.random.Generator']]=None, n_jobs: Optional[int]=None, importance_type: str='split', **kwargs):
        """Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \\*\\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError('scikit-learn is required for lightgbm.sklearn. You must install scikit-learn and restart your session to use this module.')
        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self._Booster: Optional[Booster] = None
        self._evals_result: _EvalResultDict = {}
        self._best_score: _LGBM_BoosterBestScoreType = {}
        self._best_iteration: int = -1
        self._other_params: Dict[str, Any] = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight: Optional[Union[Dict, str]] = None
        self._class_map: Optional[Dict[int, int]] = None
        self._n_features: int = -1
        self._n_features_in: int = -1
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = -1
        self.set_params(**kwargs)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'sparse', '1dlabels'], '_xfail_checks': {'check_no_attributes_set_in_init': 'scikit-learn incorrectly asserts that private attributes cannot be set in __init__: (see https://github.com/microsoft/LightGBM/issues/2628)'}}

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, 'fitted_', False)

    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params: Any) -> 'LGBMModel':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
            self._other_params[key] = value
        return self

    def _process_params(self, stage: str) -> Dict[str, Any]:
        """Process the parameters of this estimator based on its type, parameter aliases, etc.

        Parameters
        ----------
        stage : str
            Name of the stage (can be ``fit`` or ``predict``) this method is called from.

        Returns
        -------
        processed_params : dict
            Processed parameter names mapped to their values.
        """
        assert stage in {'fit', 'predict'}
        params = self.get_params()
        params.pop('objective', None)
        for alias in _ConfigAliases.get('objective'):
            if alias in params:
                obj = params.pop(alias)
                _log_warning(f"Found '{alias}' in params. Will use it instead of 'objective' argument")
                if stage == 'fit':
                    self._objective = obj
        if stage == 'fit':
            if self._objective is None:
                if isinstance(self, LGBMRegressor):
                    self._objective = 'regression'
                elif isinstance(self, LGBMClassifier):
                    if self._n_classes > 2:
                        self._objective = 'multiclass'
                    else:
                        self._objective = 'binary'
                elif isinstance(self, LGBMRanker):
                    self._objective = 'lambdarank'
                else:
                    raise ValueError('Unknown LGBMModel type.')
        if callable(self._objective):
            if stage == 'fit':
                params['objective'] = _ObjectiveFunctionWrapper(self._objective)
            else:
                params['objective'] = 'None'
        else:
            params['objective'] = self._objective
        params.pop('importance_type', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(np.iinfo(np.int32).max)
        elif isinstance(params['random_state'], np_random_Generator):
            params['random_state'] = int(params['random_state'].integers(np.iinfo(np.int32).max))
        if self._n_classes > 2:
            for alias in _ConfigAliases.get('num_class'):
                params.pop(alias, None)
            params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            eval_at = self._eval_at
            for alias in _ConfigAliases.get('eval_at'):
                if alias in params:
                    _log_warning(f"Found '{alias}' in params. Will use it instead of 'eval_at' argument")
                    eval_at = params.pop(alias)
            params['eval_at'] = eval_at
        original_metric = self._objective if isinstance(self._objective, str) else None
        if original_metric is None:
            if isinstance(self, LGBMRegressor):
                original_metric = 'l2'
            elif isinstance(self, LGBMClassifier):
                original_metric = 'multi_logloss' if self._n_classes > 2 else 'binary_logloss'
            elif isinstance(self, LGBMRanker):
                original_metric = 'ndcg'
        params = _choose_param_value('metric', params, original_metric)
        if stage == 'fit':
            params = _choose_param_value('num_threads', params, self.n_jobs)
            params['num_threads'] = self._process_n_jobs(params['num_threads'])
        return params

    def _process_n_jobs(self, n_jobs: Optional[int]) -> int:
        """Convert special values of n_jobs to their actual values according to the formulas that apply.

        Parameters
        ----------
        n_jobs : int or None
            The original value of n_jobs, potentially having special values such as 'None' or
            negative integers.

        Returns
        -------
        n_jobs : int
            The value of n_jobs with special values converted to actual number of threads.
        """
        if n_jobs is None:
            n_jobs = _LGBMCpuCount(only_physical_cores=True)
        elif n_jobs < 0:
            n_jobs = max(_LGBMCpuCount(only_physical_cores=False) + 1 + n_jobs, 1)
        return n_jobs

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, group: Optional[_LGBM_GroupType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_class_weight: Optional[List[float]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_group: Optional[List[_LGBM_GroupType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, 'LGBMModel']]=None) -> 'LGBMModel':
        """Docstring is set after definition, using a template."""
        params = self._process_params(stage='fit')
        eval_metric_list: List[Union[str, _LGBM_ScikitCustomEvalFunction]]
        if eval_metric is None:
            eval_metric_list = []
        elif isinstance(eval_metric, list):
            eval_metric_list = copy.deepcopy(eval_metric)
        else:
            eval_metric_list = [copy.deepcopy(eval_metric)]
        eval_metrics_callable = [_EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]
        params['metric'] = [params['metric']] if isinstance(params['metric'], (str, type(None))) else params['metric']
        params['metric'] = [e for e in eval_metrics_builtin if e not in params['metric']] + params['metric']
        params['metric'] = [metric for metric in params['metric'] if metric is not None]
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            _X, _y = _LGBMCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            if sample_weight is not None:
                sample_weight = _LGBMCheckSampleWeight(sample_weight, _X)
        else:
            _X, _y = (X, y)
        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)
        self._n_features = _X.shape[1]
        self._n_features_in = self._n_features
        train_set = Dataset(data=_X, label=_y, weight=sample_weight, group=group, init_score=init_score, categorical_feature=categorical_feature, params=params)
        valid_sets: List[Dataset] = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError(f'{name} should be dict or list')
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(eval_sample_weight, 'eval_sample_weight', i)
                    valid_class_weight = _get_meta_data(eval_class_weight, 'eval_class_weight', i)
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for k, v in valid_class_weight.items()}
                        valid_class_sample_weight = _LGBMComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _get_meta_data(eval_init_score, 'eval_init_score', i)
                    valid_group = _get_meta_data(eval_group, 'eval_group', i)
                    valid_set = Dataset(data=valid_data[0], label=valid_data[1], weight=valid_weight, group=valid_group, init_score=valid_init_score, categorical_feature='auto', params=params)
                valid_sets.append(valid_set)
        if isinstance(init_model, LGBMModel):
            init_model = init_model.booster_
        if callbacks is None:
            callbacks = []
        else:
            callbacks = copy.copy(callbacks)
        evals_result: _EvalResultDict = {}
        callbacks.append(record_evaluation(evals_result))
        self._Booster = train(params=params, train_set=train_set, num_boost_round=self.n_estimators, valid_sets=valid_sets, valid_names=eval_names, feval=eval_metrics_callable, init_model=init_model, feature_name=feature_name, callbacks=callbacks)
        self._evals_result = evals_result
        self._best_iteration = self._Booster.best_iteration
        self._best_score = self._Booster.best_score
        self.fitted_ = True
        self._Booster.free_dataset()
        del train_set, valid_sets
        return self
    fit.__doc__ = _lgbmmodel_doc_fit.format(X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", y_shape='numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples]', sample_weight_shape='numpy array, pandas Series, list of int or float of shape = [n_samples] or None, optional (default=None)', init_score_shape='numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)', group_shape='numpy array, pandas Series, list of int or float, or None, optional (default=None)', eval_sample_weight_shape='list of array (same types as ``sample_weight`` supports), or None, optional (default=None)', eval_init_score_shape='list of array (same types as ``init_score`` supports), or None, optional (default=None)', eval_group_shape='list of array (same types as ``group`` supports), or None, optional (default=None)') + '\n\n' + _lgbmmodel_doc_custom_eval_note

    def predict(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        """Docstring is set after definition, using a template."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('Estimator not fitted, call fit before exploiting the model.')
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError(f'Number of features of the model must match the input. Model n_features_ is {self._n_features} and input n_features is {n_features}')
        predict_params = self._process_params(stage='predict')
        for alias in _ConfigAliases.get_by_alias('data', 'X', 'raw_score', 'start_iteration', 'num_iteration', 'pred_leaf', 'pred_contrib', *kwargs.keys()):
            predict_params.pop(alias, None)
        predict_params.update(kwargs)
        predict_params = _choose_param_value('num_threads', predict_params, self.n_jobs)
        predict_params['num_threads'] = self._process_n_jobs(predict_params['num_threads'])
        return self._Booster.predict(X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **predict_params)
    predict.__doc__ = _lgbmmodel_doc_predict.format(description='Return the predicted value for each sample.', X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", output_name='predicted_result', predicted_result_shape='array-like of shape = [n_samples] or shape = [n_samples, n_classes]', X_leaves_shape='array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]', X_SHAP_values_shape='array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects')

    @property
    def n_features_(self) -> int:
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features found. Need to call fit beforehand.')
        return self._n_features

    @property
    def n_features_in_(self) -> int:
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features_in found. Need to call fit beforehand.')
        return self._n_features_in

    @property
    def best_score_(self) -> _LGBM_BoosterBestScoreType:
        """:obj:`dict`: The best score of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_score found. Need to call fit beforehand.')
        return self._best_score

    @property
    def best_iteration_(self) -> int:
        """:obj:`int`: The best iteration of fitted model if ``early_stopping()`` callback has been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_iteration found. Need to call fit with early_stopping callback beforehand.')
        return self._best_iteration

    @property
    def objective_(self) -> Union[str, _LGBM_ScikitCustomObjectiveFunction]:
        """:obj:`str` or :obj:`callable`: The concrete objective used while fitting this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No objective found. Need to call fit beforehand.')
        return self._objective

    @property
    def n_estimators_(self) -> int:
        """:obj:`int`: True number of boosting iterations performed.

        This might be less than parameter ``n_estimators`` if early stopping was enabled or
        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.
        
        .. versionadded:: 4.0.0
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_estimators found. Need to call fit beforehand.')
        return self._Booster.current_iteration()

    @property
    def n_iter_(self) -> int:
        """:obj:`int`: True number of boosting iterations performed.

        This might be less than parameter ``n_estimators`` if early stopping was enabled or
        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.
        
        .. versionadded:: 4.0.0
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_iter found. Need to call fit beforehand.')
        return self._Booster.current_iteration()

    @property
    def booster_(self) -> Booster:
        """Booster: The underlying Booster of this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self) -> _EvalResultDict:
        """:obj:`dict`: The evaluation results if validation sets have been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No results found. Need to call fit with eval_set beforehand.')
        return self._evals_result

    @property
    def feature_importances_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).

        .. note::

            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self._Booster.feature_importance(importance_type=self.importance_type)

    @property
    def feature_name_(self) -> List[str]:
        """:obj:`list` of shape = [n_features]: The names of features."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_name found. Need to call fit beforehand.')
        return self._Booster.feature_name()