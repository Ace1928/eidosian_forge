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
def _compute_builtin_metrics(self):
    """
        Helper method for computing builtin metrics
        """
    self._evaluate_sklearn_model_score_if_scorable()
    if self.model_type == _ModelType.CLASSIFIER:
        if self.is_binomial:
            self.metrics_values.update(_get_aggregate_metrics_values(_get_binary_classifier_metrics(y_true=self.y, y_pred=self.y_pred, y_proba=self.y_probs, labels=self.label_list, pos_label=self.pos_label, sample_weights=self.sample_weights)))
            self._compute_roc_and_pr_curve()
        else:
            average = self.evaluator_config.get('average', 'weighted')
            self.metrics_values.update(_get_aggregate_metrics_values(_get_multiclass_classifier_metrics(y_true=self.y, y_pred=self.y_pred, y_proba=self.y_probs, labels=self.label_list, average=average, sample_weights=self.sample_weights)))
    elif self.model_type == _ModelType.REGRESSOR:
        self.metrics_values.update(_get_aggregate_metrics_values(_get_regressor_metrics(self.y, self.y_pred, self.sample_weights)))