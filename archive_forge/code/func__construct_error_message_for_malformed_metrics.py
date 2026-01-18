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
def _construct_error_message_for_malformed_metrics(self, malformed_results, input_columns, output_columns):
    error_messages = [self._get_error_message_missing_columns(metric_name, param_names) for metric_name, param_names in malformed_results]
    joined_error_message = '\n'.join(error_messages)
    full_message = f"Error: Metric calculation failed for the following metrics:\n        {joined_error_message}\n\n        Below are the existing column names for the input/output data:\n        Input Columns: {input_columns}\n        Output Columns: {output_columns}\n\n        To resolve this issue, you may need to:\n         - specify any required parameters\n         - if you are missing columns, check that there are no circular dependencies among your\n         metrics, and you may want to map them to an existing column using the following\n         configuration:\n        evaluator_config={{'col_mapping': {{<missing column name>: <existing column name>}}}}"
    return '\n'.join((l.lstrip() for l in full_message.splitlines()))