import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _autolog(flavor_name=FLAVOR_NAME, log_input_examples=False, log_model_signatures=True, log_models=True, log_datasets=True, disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False, max_tuning_runs=5, log_post_training_metrics=True, serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE, pos_label=None, extra_tags=None):
    """
    Internal autologging function for scikit-learn models.

    Args:
        flavor_name: A string value. Enable a ``mlflow.sklearn`` autologging routine
            for a flavor. By default it enables autologging for original
            scikit-learn models, as ``mlflow.sklearn.autolog()`` does. If
            the argument is `xgboost`, autologging for XGBoost scikit-learn
            models is enabled.
    """
    import pandas as pd
    import sklearn
    import sklearn.metrics
    import sklearn.model_selection
    from mlflow.models import infer_signature
    from mlflow.sklearn.utils import _TRAINING_PREFIX, _create_child_runs_for_parameter_search, _gen_lightgbm_sklearn_estimators_to_patch, _gen_xgboost_sklearn_estimators_to_patch, _get_estimator_info_tags, _get_X_y_and_sample_weight, _is_parameter_search_estimator, _log_estimator_content, _log_parameter_search_results_as_artifact
    from mlflow.tracking.context import registry as context_registry
    if max_tuning_runs is not None and max_tuning_runs < 0:
        raise MlflowException(message=f'`max_tuning_runs` must be non-negative, instead got {max_tuning_runs}.', error_code=INVALID_PARAMETER_VALUE)

    def fit_mlflow_xgboost_and_lightgbm(original, self, *args, **kwargs):
        """
        Autologging function for XGBoost and LightGBM scikit-learn models
        """
        input_example_exc = None
        try:
            input_example = deepcopy(_get_X_y_and_sample_weight(self.fit, args, kwargs)[0][:INPUT_EXAMPLE_SAMPLE_ROWS])
        except Exception as e:
            input_example_exc = e

        def get_input_example():
            if input_example_exc is not None:
                raise input_example_exc
            else:
                return input_example
        fit_output = original(self, *args, **kwargs)
        if log_models:
            input_example, signature = resolve_input_example_and_signature(get_input_example, lambda input_example: infer_signature(input_example, self.predict(deepcopy(input_example))), log_input_examples, log_model_signatures, _logger)
            log_model_func = mlflow.xgboost.log_model if flavor_name == mlflow.xgboost.FLAVOR_NAME else mlflow.lightgbm.log_model
            registered_model_name = get_autologging_config(flavor_name, 'registered_model_name', None)
            if flavor_name == mlflow.xgboost.FLAVOR_NAME:
                model_format = get_autologging_config(flavor_name, 'model_format', 'xgb')
                log_model_func(self, artifact_path='model', signature=signature, input_example=input_example, registered_model_name=registered_model_name, model_format=model_format)
            else:
                log_model_func(self, artifact_path='model', signature=signature, input_example=input_example, registered_model_name=registered_model_name)
        return fit_output

    def fit_mlflow(original, self, *args, **kwargs):
        """
        Autologging function that performs model training by executing the training method
        referred to be `func_name` on the instance of `clazz` referred to by `self` & records
        MLflow parameters, metrics, tags, and artifacts to a corresponding MLflow Run.
        """
        X, y_true, sample_weight = _get_X_y_and_sample_weight(self.fit, args, kwargs)
        autologging_client = MlflowAutologgingQueueingClient()
        _log_pretraining_metadata(autologging_client, self, X, y_true)
        params_logging_future = autologging_client.flush(synchronous=False)
        fit_output = original(self, *args, **kwargs)
        _log_posttraining_metadata(autologging_client, self, X, y_true, sample_weight)
        autologging_client.flush(synchronous=True)
        params_logging_future.await_completion()
        return fit_output

    def _log_pretraining_metadata(autologging_client, estimator, X, y):
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        Args:
            autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                efficiently logging run data to MLflow Tracking.
            estimator: The scikit-learn estimator for which to log metadata.
        """
        should_log_params_deeply = not _is_parameter_search_estimator(estimator)
        run_id = mlflow.active_run().info.run_id
        autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=estimator.get_params(deep=should_log_params_deeply))
        autologging_client.set_tags(run_id=run_id, tags=_get_estimator_info_tags(estimator))
        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(context_tags)
                dataset = _create_dataset(X, source, y)
                if dataset:
                    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='train')]
                    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                    autologging_client.log_inputs(run_id=mlflow.active_run().info.run_id, datasets=[dataset_input])
            except Exception as e:
                _logger.warning('Failed to log training dataset information to MLflow Tracking. Reason: %s', e)

    def _log_posttraining_metadata(autologging_client, estimator, X, y, sample_weight):
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        Args:
            autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                efficiently logging run data to MLflow Tracking.
            estimator: The scikit-learn estimator for which to log metadata.
            X: The training dataset samples passed to the ``estimator.fit()`` function.
            y: The training dataset labels passed to the ``estimator.fit()`` function.
            sample_weight: Sample weights passed to the ``estimator.fit()`` function.
        """
        input_example_exc = None
        try:
            input_example = deepcopy(X[:INPUT_EXAMPLE_SAMPLE_ROWS])
        except Exception as e:
            input_example_exc = e

        def get_input_example():
            if input_example_exc is not None:
                raise input_example_exc
            else:
                return input_example

        def infer_model_signature(input_example):
            if hasattr(estimator, 'predict'):
                model_output = estimator.predict(deepcopy(input_example))
            elif hasattr(estimator, 'transform'):
                model_output = estimator.transform(deepcopy(input_example))
            else:
                raise Exception('the trained model does not have a `predict` or `transform` function, which is required in order to infer the signature')
            return infer_signature(input_example, model_output)
        logged_metrics = _log_estimator_content(autologging_client=autologging_client, estimator=estimator, prefix=_TRAINING_PREFIX, run_id=mlflow.active_run().info.run_id, X=X, y_true=y, sample_weight=sample_weight, pos_label=pos_label)
        if y is None and (not logged_metrics):
            _logger.warning('Training metrics will not be recorded because training labels were not specified. To automatically record training metrics, provide training labels as inputs to the model training function.')

        def _log_model_with_except_handling(*args, **kwargs):
            try:
                return log_model(*args, **kwargs)
            except _SklearnCustomModelPicklingError as e:
                _logger.warning(str(e))
        if log_models:
            input_example, signature = resolve_input_example_and_signature(get_input_example, infer_model_signature, log_input_examples, log_model_signatures, _logger)
            registered_model_name = get_autologging_config(FLAVOR_NAME, 'registered_model_name', None)
            _log_model_with_except_handling(estimator, artifact_path='model', signature=signature, input_example=input_example, serialization_format=serialization_format, registered_model_name=registered_model_name)
        if _is_parameter_search_estimator(estimator):
            if hasattr(estimator, 'best_estimator_') and log_models:
                _log_model_with_except_handling(estimator.best_estimator_, artifact_path='best_estimator', signature=signature, input_example=input_example, serialization_format=serialization_format)
            if hasattr(estimator, 'best_score_'):
                autologging_client.log_metrics(run_id=mlflow.active_run().info.run_id, metrics={'best_cv_score': estimator.best_score_})
            if hasattr(estimator, 'best_params_'):
                best_params = {f'best_{param_name}': param_value for param_name, param_value in estimator.best_params_.items()}
                autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=best_params)
            if hasattr(estimator, 'cv_results_'):
                try:
                    child_tags = context_registry.resolve_tags()
                    child_tags.update({MLFLOW_AUTOLOGGING: flavor_name})
                    _create_child_runs_for_parameter_search(autologging_client=autologging_client, cv_estimator=estimator, parent_run=mlflow.active_run(), max_tuning_runs=max_tuning_runs, child_tags=child_tags)
                except Exception as e:
                    _logger.warning(f'Encountered exception during creation of child runs for parameter search. Child runs may be missing. Exception: {e}')
                try:
                    cv_results_df = pd.DataFrame.from_dict(estimator.cv_results_)
                    _log_parameter_search_results_as_artifact(cv_results_df, mlflow.active_run().info.run_id)
                except Exception as e:
                    _logger.warning(f'Failed to log parameter search results as an artifact. Exception: {e}')

    def patched_fit(fit_impl, allow_children_patch, original, self, *args, **kwargs):
        """
        Autologging patch function to be applied to a sklearn model class that defines a `fit`
        method and inherits from `BaseEstimator` (thereby defining the `get_params()` method)

        Args:
            fit_impl: The patched fit function implementation, the function should be defined as
                `fit_mlflow(original, self, *args, **kwargs)`, the `original` argument
                refers to the original `EstimatorClass.fit` method, the `self` argument
                refers to the estimator instance being patched, the `*args` and
                `**kwargs` are arguments passed to the original fit method.
            allow_children_patch: Whether to allow children sklearn session logging or not.
            original: the original `EstimatorClass.fit` method to be patched.
            self: the estimator instance being patched.
            args: positional arguments to be passed to the original fit method.
            kwargs: keyword arguments to be passed to the original fit method.
        """
        should_log_post_training_metrics = log_post_training_metrics and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        with _SklearnTrainingSession(estimator=self, allow_children=allow_children_patch) as t:
            if t.should_log():
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    result = fit_impl(original, self, *args, **kwargs)
                if should_log_post_training_metrics:
                    _AUTOLOGGING_METRICS_MANAGER.register_model(self, mlflow.active_run().info.run_id)
                return result
            else:
                return original(self, *args, **kwargs)

    def patched_predict(original, self, *args, **kwargs):
        """
        In `patched_predict`, register the prediction result instance with the run id and
         eval dataset name. e.g.
        ```
        prediction_result = model_1.predict(eval_X)
        ```
        then we need register the following relationship into the `_AUTOLOGGING_METRICS_MANAGER`:
        id(prediction_result) --> (eval_dataset_name, run_id)

        Note: we cannot set additional attributes "eval_dataset_name" and "run_id" into
        the prediction_result object, because certain dataset type like numpy does not support
        additional attribute assignment.
        """
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                predict_result = original(self, *args, **kwargs)
            eval_dataset = get_instance_method_first_arg_value(original, args, kwargs)
            eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(self, eval_dataset)
            _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(run_id, eval_dataset_name, predict_result)
            if log_datasets:
                try:
                    context_tags = context_registry.resolve_tags()
                    source = CodeDatasetSource(context_tags)
                    dataset = _create_dataset(eval_dataset, source)
                    if dataset:
                        tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='eval')]
                        dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                        client = mlflow.MlflowClient()
                        client.log_inputs(run_id=run_id, datasets=[dataset_input])
                except Exception as e:
                    _logger.warning('Failed to log evaluation dataset information to MLflow Tracking. Reason: %s', e)
            return predict_result
        else:
            return original(self, *args, **kwargs)

    def patched_metric_api(original, *args, **kwargs):
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                metric = original(*args, **kwargs)
            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
                metric_name = original.__name__
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(None, original, *args, **kwargs)
                run_id, dataset_name = _AUTOLOGGING_METRICS_MANAGER.get_run_id_and_dataset_name_for_metric_api_call(args, kwargs)
                if run_id and dataset_name:
                    metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(run_id, metric_name, dataset_name, call_command)
                    _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(run_id, metric_key, metric)
            return metric
        else:
            return original(*args, **kwargs)

    def patched_model_score(original, self, *args, **kwargs):
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                score_value = original(self, *args, **kwargs)
            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(score_value):
                metric_name = f'{self.__class__.__name__}_score'
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(self, original, *args, **kwargs)
                eval_dataset = get_instance_method_first_arg_value(original, args, kwargs)
                eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(self, eval_dataset)
                metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(run_id, metric_name, eval_dataset_name, call_command)
                _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(run_id, metric_key, score_value)
            return score_value
        else:
            return original(self, *args, **kwargs)

    def _apply_sklearn_descriptor_unbound_method_call_fix():
        import sklearn
        if Version(sklearn.__version__) <= Version('0.24.2'):
            import sklearn.utils.metaestimators
            if not hasattr(sklearn.utils.metaestimators, '_IffHasAttrDescriptor'):
                return

            def patched_IffHasAttrDescriptor__get__(self, obj, type=None):
                """
                For sklearn version <= 0.24.2, `_IffHasAttrDescriptor.__get__` method does not
                support unbound method call.
                See https://github.com/scikit-learn/scikit-learn/issues/20614
                This patched function is for hot patch.
                """
                if obj is not None:
                    for delegate_name in self.delegate_names:
                        try:
                            delegate = sklearn.utils.metaestimators.attrgetter(delegate_name)(obj)
                        except AttributeError:
                            continue
                        else:
                            getattr(delegate, self.attribute_name)
                            break
                    else:
                        sklearn.utils.metaestimators.attrgetter(self.delegate_names[-1])(obj)

                    def out(*args, **kwargs):
                        return self.fn(obj, *args, **kwargs)
                else:

                    def out(*args, **kwargs):
                        return self.fn(*args, **kwargs)
                functools.update_wrapper(out, self.fn)
                return out
            update_wrapper_extended(patched_IffHasAttrDescriptor__get__, sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__)
            sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__ = patched_IffHasAttrDescriptor__get__
    _apply_sklearn_descriptor_unbound_method_call_fix()
    if flavor_name == mlflow.xgboost.FLAVOR_NAME:
        estimators_to_patch = _gen_xgboost_sklearn_estimators_to_patch()
        patched_fit_impl = fit_mlflow_xgboost_and_lightgbm
        allow_children_patch = True
    elif flavor_name == mlflow.lightgbm.FLAVOR_NAME:
        estimators_to_patch = _gen_lightgbm_sklearn_estimators_to_patch()
        patched_fit_impl = fit_mlflow_xgboost_and_lightgbm
        allow_children_patch = True
    else:
        estimators_to_patch = _gen_estimators_to_patch()
        patched_fit_impl = fit_mlflow
        allow_children_patch = False
    for class_def in estimators_to_patch:
        for func_name in ['fit', 'fit_transform', 'fit_predict']:
            _patch_estimator_method_if_available(flavor_name, class_def, func_name, functools.partial(patched_fit, patched_fit_impl, allow_children_patch), manage_run=True, extra_tags=extra_tags)
        for func_name in ['predict', 'predict_proba', 'transform', 'predict_log_proba']:
            _patch_estimator_method_if_available(flavor_name, class_def, func_name, patched_predict, manage_run=False)
        _patch_estimator_method_if_available(flavor_name, class_def, 'score', patched_model_score, manage_run=False, extra_tags=extra_tags)
    if log_post_training_metrics:
        for metric_name in _get_metric_name_list():
            safe_patch(flavor_name, sklearn.metrics, metric_name, patched_metric_api, manage_run=False)
        if hasattr(sklearn.metrics, 'get_scorer_names'):
            for scoring in sklearn.metrics.get_scorer_names():
                scorer = sklearn.metrics.get_scorer(scoring)
                safe_patch(flavor_name, scorer, '_score_func', patched_metric_api, manage_run=False)
        else:
            for scorer in sklearn.metrics.SCORERS.values():
                safe_patch(flavor_name, scorer, '_score_func', patched_metric_api, manage_run=False)

    def patched_fn_with_autolog_disabled(original, *args, **kwargs):
        with disable_autologging():
            return original(*args, **kwargs)
    for disable_autolog_func_name in _apis_autologging_disabled:
        safe_patch(flavor_name, sklearn.model_selection, disable_autolog_func_name, patched_fn_with_autolog_disabled, manage_run=False)

    def _create_dataset(X, source, y=None, dataset_name=None):
        from scipy.sparse import issparse
        if isinstance(X, pd.DataFrame):
            dataset = from_pandas(df=X, source=source)
        elif issparse(X):
            arr_X = X.toarray()
            if y is not None:
                arr_y = y.toarray()
                dataset = from_numpy(features=arr_X, targets=arr_y, source=source, name=dataset_name)
            else:
                dataset = from_numpy(features=arr_X, source=source, name=dataset_name)
        elif isinstance(X, np.ndarray):
            if y is not None:
                dataset = from_numpy(features=X, targets=y, source=source, name=dataset_name)
            else:
                dataset = from_numpy(features=X, source=source, name=dataset_name)
        else:
            _logger.warning('Unrecognized dataset type %s. Dataset logging skipped.', type(X))
            return None
        return dataset