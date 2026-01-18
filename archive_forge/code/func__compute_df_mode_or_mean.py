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
def _compute_df_mode_or_mean(df):
    """
    Compute mean (for continuous columns) and compute mode (for other columns) for the
    input dataframe, return a dict, key is column name, value is the corresponding mode or
    mean value, this function calls `_is_continuous` to determine whether the
    column is continuous column.
    """
    continuous_cols = [c for c in df.columns if _is_continuous(df[c])]
    df_cont = df[continuous_cols]
    df_non_cont = df.drop(continuous_cols, axis=1)
    means = {} if df_cont.empty else df_cont.mean().to_dict()
    modes = {} if df_non_cont.empty else df_non_cont.mode().loc[0].to_dict()
    return {**means, **modes}