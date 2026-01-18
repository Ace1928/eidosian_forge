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
def _log_specialized_estimator_content(autologging_client, fitted_estimator, run_id, prefix, X, y_true, sample_weight, pos_label):
    import sklearn
    metrics = {}
    if y_true is not None:
        try:
            if sklearn.base.is_classifier(fitted_estimator):
                metrics = _get_classifier_metrics(fitted_estimator, prefix, X, y_true, sample_weight, pos_label)
            elif sklearn.base.is_regressor(fitted_estimator):
                metrics = _get_regressor_metrics(fitted_estimator, prefix, X, y_true, sample_weight)
        except Exception as err:
            msg = 'Failed to autolog metrics for ' + fitted_estimator.__class__.__name__ + '. Logging error: ' + str(err)
            _logger.warning(msg)
        else:
            autologging_client.log_metrics(run_id=run_id, metrics=metrics)
    if sklearn.base.is_classifier(fitted_estimator):
        try:
            artifacts = _get_classifier_artifacts(fitted_estimator, prefix, X, y_true, sample_weight)
        except Exception as e:
            msg = 'Failed to autolog artifacts for ' + fitted_estimator.__class__.__name__ + '. Logging error: ' + str(e)
            _logger.warning(msg)
            return metrics
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as ie:
            _logger.warning(f'Failed to import matplotlib (error: {ie!r}). Skipping artifact logging.')
            return metrics
        _matplotlib_config = {'savefig.dpi': 175, 'figure.autolayout': True, 'font.size': 8}
        with TempDir() as tmp_dir:
            for artifact in artifacts:
                try:
                    with matplotlib.rc_context(_matplotlib_config):
                        display = artifact.function(**artifact.arguments)
                        display.ax_.set_title(artifact.title)
                        artifact_path = f'{artifact.name}.png'
                        filepath = tmp_dir.path(artifact_path)
                        display.figure_.savefig(fname=filepath, format='png')
                        plt.close(display.figure_)
                except Exception as e:
                    _log_warning_for_artifacts(artifact.name, artifact.function, e)
            MlflowClient().log_artifacts(run_id, tmp_dir.path())
    return metrics