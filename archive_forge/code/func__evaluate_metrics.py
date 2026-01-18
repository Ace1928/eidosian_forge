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
def _evaluate_metrics(self, eval_df):
    self._order_extra_metrics(eval_df)
    self._test_first_row(eval_df)
    input_df = self.X.copy_to_avoid_mutation()
    for metric_tuple in self.ordered_metrics:
        _, eval_fn_args = self._get_args_for_metrics(metric_tuple, eval_df, input_df)
        metric_value = _evaluate_metric(metric_tuple, eval_fn_args)
        if metric_value:
            name = f'{metric_tuple.name}/{metric_tuple.version}' if metric_tuple.version else metric_tuple.name
            self.metrics_values.update({name: metric_value})