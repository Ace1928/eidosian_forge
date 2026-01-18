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
def _log_eval_table(self):
    if not any((metric_value.scores is not None or metric_value.justifications is not None for _, metric_value in self.metrics_values.items())):
        return
    metric_prefix = self.evaluator_config.get('metric_prefix', '')
    if not isinstance(metric_prefix, str):
        metric_prefix = ''
    if isinstance(self.dataset.features_data, pd.DataFrame):
        if self.dataset.has_targets:
            data = self.dataset.features_data.assign(**{self.dataset.targets_name or 'target': self.y, self.dataset.predictions_name or self.predictions or 'outputs': self.y_pred})
        else:
            data = self.dataset.features_data.assign(outputs=self.y_pred)
    else:
        data = pd.DataFrame(self.dataset.features_data, columns=self.dataset.feature_names)
        if self.dataset.has_targets:
            data = data.assign(**{self.dataset.targets_name or 'target': self.y, self.dataset.predictions_name or self.predictions or 'outputs': self.y_pred})
        else:
            data = data.assign(outputs=self.y_pred)
    if self.other_output_columns is not None and len(self.other_output_columns_for_eval) > 0:
        for column in self.other_output_columns_for_eval:
            data[column] = self.other_output_columns[column]
    columns = {}
    for metric_name, metric_value in self.metrics_values.items():
        scores = metric_value.scores
        justifications = metric_value.justifications
        if scores:
            if metric_name.startswith(metric_prefix) and metric_name[len(metric_prefix):] in [_TOKEN_COUNT_METRIC_NAME, _LATENCY_METRIC_NAME]:
                columns[metric_name] = scores
            else:
                columns[f'{metric_name}/score'] = scores
        if justifications:
            columns[f'{metric_name}/justification'] = justifications
    data = data.assign(**columns)
    artifact_file_name = f'{metric_prefix}{_EVAL_TABLE_FILE_NAME}'
    mlflow.log_table(data, artifact_file=artifact_file_name)
    if self.eval_results_path:
        eval_table_spark = self.spark_session.createDataFrame(data)
        try:
            eval_table_spark.write.mode(self.eval_results_mode).option('mergeSchema', 'true').format('delta').saveAsTable(self.eval_results_path)
        except Exception as e:
            _logger.info(f'Saving eval table to delta table failed. Reason: {e}')
    name = _EVAL_TABLE_FILE_NAME.split('.', 1)[0]
    self.artifacts[name] = JsonEvaluationArtifact(uri=mlflow.get_artifact_uri(artifact_file_name))