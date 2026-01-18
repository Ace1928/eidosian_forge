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
def _get_classifier_artifacts(fitted_estimator, prefix, X, y_true, sample_weight):
    """
    Draw and record various common artifacts for classifier

    For all classifiers, we always log:
    (1) confusion matrix:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

    For only binary classifiers, we will log:
    (2) precision recall curve:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html
    (3) roc curve:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and split into train & test datasets.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a list of artifacts path to be logged

    Args:
        fitted_estimator: The already fitted regressor
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        List of artifacts to be logged
    """
    import sklearn
    if not _is_plotting_supported():
        return []
    is_plot_function_deprecated = Version(sklearn.__version__) >= Version('1.0')

    def plot_confusion_matrix(*args, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt
        class_labels = _get_class_labels_from_estimator(fitted_estimator)
        if class_labels is None:
            class_labels = set(y_true)
        with matplotlib.rc_context({'font.size': min(8.0, 50.0 / len(class_labels)), 'axes.labelsize': 8.0, 'figure.dpi': 175}):
            _, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
            return sklearn.metrics.ConfusionMatrixDisplay.from_estimator(*args, **kwargs, ax=ax) if is_plot_function_deprecated else sklearn.metrics.plot_confusion_matrix(*args, **kwargs, ax=ax)
    y_true_arg_name = 'y' if is_plot_function_deprecated else 'y_true'
    classifier_artifacts = [_SklearnArtifact(name=prefix + 'confusion_matrix', function=plot_confusion_matrix, arguments=dict(estimator=fitted_estimator, X=X, sample_weight=sample_weight, normalize='true', cmap='Blues', **{y_true_arg_name: y_true}), title='Normalized confusion matrix')]
    if len(set(y_true)) == 2:
        classifier_artifacts.extend([_SklearnArtifact(name=prefix + 'roc_curve', function=sklearn.metrics.RocCurveDisplay.from_estimator if is_plot_function_deprecated else sklearn.metrics.plot_roc_curve, arguments={'estimator': fitted_estimator, 'X': X, 'y': y_true, 'sample_weight': sample_weight}, title='ROC curve'), _SklearnArtifact(name=prefix + 'precision_recall_curve', function=sklearn.metrics.PrecisionRecallDisplay.from_estimator if is_plot_function_deprecated else sklearn.metrics.plot_precision_recall_curve, arguments={'estimator': fitted_estimator, 'X': X, 'y': y_true, 'sample_weight': sample_weight}, title='Precision recall curve')])
    return classifier_artifacts