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
def _get_dataframe_with_renamed_columns(x, new_column_names):
    """
    Downstream inference functions may expect a pd.DataFrame to be created from x. However,
    if x is already a pd.DataFrame, and new_column_names != x.columns, we cannot simply call
    pd.DataFrame(x, columns=new_column_names) because the resulting pd.DataFrame will contain
    NaNs for every column in new_column_names that does not exist in x.columns. This function
    instead creates a new pd.DataFrame object from x, and then explicitly renames the columns
    to avoid NaNs.

    Args:
        x: A data object, such as a Pandas DataFrame, numPy array, or list
        new_column_names: Column names for the output Pandas DataFrame

    Returns:
        A pd.DataFrame with x as data, with columns new_column_names
    """
    df = pd.DataFrame(x)
    return df.rename(columns=dict(zip(df.columns, new_column_names)))