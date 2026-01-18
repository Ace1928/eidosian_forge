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
def _log_custom_artifacts(self, eval_df):
    if not self.custom_artifacts:
        return
    for index, custom_artifact in enumerate(self.custom_artifacts):
        with tempfile.TemporaryDirectory() as artifacts_dir:
            custom_artifact_tuple = _CustomArtifact(function=custom_artifact, index=index, name=getattr(custom_artifact, '__name__', repr(custom_artifact)), artifacts_dir=artifacts_dir)
            artifact_results = _evaluate_custom_artifacts(custom_artifact_tuple, eval_df.copy(), copy.deepcopy(self.metrics_values))
            if artifact_results:
                for artifact_name, raw_artifact in artifact_results.items():
                    self.artifacts[artifact_name] = self._log_custom_metric_artifact(artifact_name, raw_artifact, custom_artifact_tuple)