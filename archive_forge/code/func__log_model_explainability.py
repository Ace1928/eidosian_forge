import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def _log_model_explainability(self):
    if not self.evaluator_config.get('log_model_explainability', True):
        return
    if self.is_model_server and (not self.evaluator_config.get('log_model_explainability', False)):
        _logger.warning('Skipping model explainability because a model server is used for environment restoration.')
        return
    if self.model_loader_module == 'mlflow.spark':
        _logger.warning('Logging model explainability insights is not currently supported for PySpark models.')
        return
    if not (np.issubdtype(self.y.dtype, np.number) or self.y.dtype == np.bool_):
        _logger.warning('Skip logging model explainability insights because it requires all label values to be numeric or boolean.')
        return
    algorithm = self.evaluator_config.get('explainability_algorithm', None)
    if algorithm is not None and algorithm not in _SUPPORTED_SHAP_ALGORITHMS:
        raise MlflowException(message=f'Specified explainer algorithm {algorithm} is unsupported. Currently only support {','.join(_SUPPORTED_SHAP_ALGORITHMS)} algorithms.', error_code=INVALID_PARAMETER_VALUE)
    if algorithm != 'kernel':
        feature_dtypes = list(self.X.get_original().dtypes)
        for feature_dtype in feature_dtypes:
            if not np.issubdtype(feature_dtype, np.number):
                _logger.warning(f'Skip logging model explainability insights because the shap explainer {algorithm} requires all feature values to be numeric, and each feature column must only contain scalar values.')
                return
    try:
        import shap
        from matplotlib import pyplot
    except ImportError:
        _logger.warning('SHAP or matplotlib package is not installed, so model explainability insights will not be logged.')
        return
    if Version(shap.__version__) < Version('0.40'):
        _logger.warning('Shap package version is lower than 0.40, Skip log model explainability.')
        return
    is_multinomial_classifier = self.model_type == _ModelType.CLASSIFIER and self.num_classes > 2
    sample_rows = self.evaluator_config.get('explainability_nsamples', _DEFAULT_SAMPLE_ROWS_FOR_SHAP)
    X_df = self.X.copy_to_avoid_mutation()
    sampled_X = shap.sample(X_df, sample_rows, random_state=0)
    mode_or_mean_dict = _compute_df_mode_or_mean(X_df)
    sampled_X = sampled_X.fillna(mode_or_mean_dict)
    shap_predict_fn = functools.partial(_shap_predict_fn, predict_fn=self.predict_fn, feature_names=self.feature_names)
    try:
        if algorithm:
            if algorithm == 'kernel':
                from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer
                kernel_link = self.evaluator_config.get('explainability_kernel_link', 'identity')
                if kernel_link not in ['identity', 'logit']:
                    raise ValueError(f"explainability_kernel_link config can only be set to 'identity' or 'logit', but got '{kernel_link}'.")
                background_X = shap.sample(X_df, sample_rows, random_state=3)
                background_X = background_X.fillna(mode_or_mean_dict)
                explainer = _PatchedKernelExplainer(shap_predict_fn, background_X, link=kernel_link)
            else:
                explainer = shap.Explainer(shap_predict_fn, sampled_X, feature_names=self.feature_names, algorithm=algorithm)
        elif self.raw_model and (not is_multinomial_classifier) and (not isinstance(self.raw_model, sk_Pipeline)):
            explainer = shap.Explainer(self.raw_model, sampled_X, feature_names=self.feature_names)
        else:
            explainer = shap.Explainer(shap_predict_fn, sampled_X, feature_names=self.feature_names)
        _logger.info(f'Shap explainer {explainer.__class__.__name__} is used.')
        if algorithm == 'kernel':
            shap_values = shap.Explanation(explainer.shap_values(sampled_X), feature_names=self.feature_names)
        else:
            shap_values = explainer(sampled_X)
    except Exception as e:
        if not self.evaluator_config.get('ignore_exceptions', True):
            raise e
        _logger.warning(f'Shap evaluation failed. Reason: {e!r}. Set logging level to DEBUG to see the full traceback.')
        _logger.debug('', exc_info=True)
        return
    try:
        mlflow.shap.log_explainer(explainer, artifact_path='explainer')
    except Exception as e:
        _logger.warning(f'Logging explainer failed. Reason: {e!r}. Set logging level to DEBUG to see the full traceback.')
        _logger.debug('', exc_info=True)

    def _adjust_color_bar():
        pyplot.gcf().axes[-1].set_aspect('auto')
        pyplot.gcf().axes[-1].set_box_aspect(50)

    def _adjust_axis_tick():
        pyplot.xticks(fontsize=10)
        pyplot.yticks(fontsize=10)

    def plot_beeswarm():
        shap.plots.beeswarm(shap_values, show=False, color_bar=True)
        _adjust_color_bar()
        _adjust_axis_tick()
    self._log_image_artifact(plot_beeswarm, 'shap_beeswarm_plot')

    def plot_summary():
        shap.summary_plot(shap_values, show=False, color_bar=True)
        _adjust_color_bar()
        _adjust_axis_tick()
    self._log_image_artifact(plot_summary, 'shap_summary_plot')

    def plot_feature_importance():
        shap.plots.bar(shap_values, show=False)
        _adjust_axis_tick()
    self._log_image_artifact(plot_feature_importance, 'shap_feature_importance_plot')