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
def _get_binary_classifier_metrics(*, y_true, y_pred, y_proba=None, labels=None, pos_label=1, sample_weights=None):
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    return {'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn, 'true_positives': tp, **_get_common_classifier_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, labels=labels, average='binary', pos_label=pos_label, sample_weights=sample_weights)}