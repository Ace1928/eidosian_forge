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
class _MutationGuardedData:
    """
        Wrapper around a data object that requires explicit API calls to obtain either a copy
        of the data object, or, in cases where the caller can guaranteed that the object will not
        be mutated, the original data object.
        """

    def __init__(self, data):
        """
            Args:
                data: A data object, such as a Pandas DataFrame, numPy array, or list.
            """
        self._data = data

    def copy_to_avoid_mutation(self):
        """
            Obtain a copy of the data. This method should be called every time the data needs
            to be used in a context where it may be subsequently mutated, guarding against
            accidental reuse after mutation.

            Returns:
                A copy of the data object.
            """
        if isinstance(self._data, pd.DataFrame):
            return self._data.copy(deep=True)
        else:
            return copy.deepcopy(self._data)

    def get_original(self):
        """
            Obtain the original data object. This method should only be called if the caller
            can guarantee that it will not mutate the data during subsequent operations.

            Returns:
                The original data object.
            """
        return self._data