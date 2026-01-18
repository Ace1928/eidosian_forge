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
def _log_multiclass_classifier_artifacts(self):
    per_class_metrics_collection_df = _get_classifier_per_class_metrics_collection_df(self.y, self.y_pred, labels=self.label_list, sample_weights=self.sample_weights)
    log_roc_pr_curve = False
    if self.y_probs is not None:
        max_classes_for_multiclass_roc_pr = self.evaluator_config.get('max_classes_for_multiclass_roc_pr', 10)
        if self.num_classes <= max_classes_for_multiclass_roc_pr:
            log_roc_pr_curve = True
        else:
            _logger.warning(f"The classifier num_classes > {max_classes_for_multiclass_roc_pr}, skip logging ROC curve and Precision-Recall curve. You can add evaluator config 'max_classes_for_multiclass_roc_pr' to increase the threshold.")
    if log_roc_pr_curve:
        roc_curve = _gen_classifier_curve(is_binomial=False, y=self.y, y_probs=self.y_probs, labels=self.label_list, pos_label=self.pos_label, curve_type='roc', sample_weights=self.sample_weights)

        def plot_roc_curve():
            roc_curve.plot_fn(**roc_curve.plot_fn_args)
        self._log_image_artifact(plot_roc_curve, 'roc_curve_plot')
        per_class_metrics_collection_df['roc_auc'] = roc_curve.auc
        pr_curve = _gen_classifier_curve(is_binomial=False, y=self.y, y_probs=self.y_probs, labels=self.label_list, pos_label=self.pos_label, curve_type='pr', sample_weights=self.sample_weights)

        def plot_pr_curve():
            pr_curve.plot_fn(**pr_curve.plot_fn_args)
        self._log_image_artifact(plot_pr_curve, 'precision_recall_curve_plot')
        per_class_metrics_collection_df['precision_recall_auc'] = pr_curve.auc
    self._log_pandas_df_artifact(per_class_metrics_collection_df, 'per_class_metrics')