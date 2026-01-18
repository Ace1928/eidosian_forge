import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter
import numpy as np
from packaging.version import Version
from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
def _get_classifier_metrics(fitted_estimator, prefix, X, y_true, sample_weight, pos_label):
    """
    Compute and record various common metrics for classifiers

    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, when `pos_label` is not specified (passed in as `None`), we set `average`
    to `weighted` to compute the weighted score of these metrics.
    When the `pos_label` is specified (not `None`), we set `average` to `binary`.

    For (4) accuracy score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy,
    as opposed to `False` that outputs the absolute correct number of sample prediction

    We log additional metrics if certain classifier has method `predict_proba`
    (5) log loss:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    (6) roc_auc_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    By default, for roc_auc_score, we pick `average` to be `weighted`, `multi_class` to be `ovo`,
    to make the output more insensitive to dataset imbalance.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, ...... sample_weight), otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)

    Args:
        fitted_estimator: The already fitted classifier
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        dictionary of (function name, computed value)
    """
    import sklearn
    average = 'weighted' if pos_label is None else 'binary'
    y_pred = fitted_estimator.predict(X)
    classifier_metrics = [_SklearnMetric(name=prefix + 'precision_score', function=sklearn.metrics.precision_score, arguments={'y_true': y_true, 'y_pred': y_pred, 'pos_label': pos_label, 'average': average, 'sample_weight': sample_weight}), _SklearnMetric(name=prefix + 'recall_score', function=sklearn.metrics.recall_score, arguments={'y_true': y_true, 'y_pred': y_pred, 'pos_label': pos_label, 'average': average, 'sample_weight': sample_weight}), _SklearnMetric(name=prefix + 'f1_score', function=sklearn.metrics.f1_score, arguments={'y_true': y_true, 'y_pred': y_pred, 'pos_label': pos_label, 'average': average, 'sample_weight': sample_weight}), _SklearnMetric(name=prefix + 'accuracy_score', function=sklearn.metrics.accuracy_score, arguments={'y_true': y_true, 'y_pred': y_pred, 'normalize': True, 'sample_weight': sample_weight})]
    if hasattr(fitted_estimator, 'predict_proba'):
        y_pred_proba = fitted_estimator.predict_proba(X)
        classifier_metrics.extend([_SklearnMetric(name=prefix + 'log_loss', function=sklearn.metrics.log_loss, arguments={'y_true': y_true, 'y_pred': y_pred_proba, 'sample_weight': sample_weight})])
        if _is_metric_supported('roc_auc_score'):
            if len(y_pred_proba[0]) == 2:
                y_pred_proba = y_pred_proba[:, 1]
            classifier_metrics.extend([_SklearnMetric(name=prefix + 'roc_auc', function=sklearn.metrics.roc_auc_score, arguments={'y_true': y_true, 'y_score': y_pred_proba, 'average': 'weighted', 'sample_weight': sample_weight, 'multi_class': 'ovo'})])
    return _get_metrics_value_dict(classifier_metrics)