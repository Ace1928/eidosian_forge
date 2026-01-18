import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
class _ServedPyFuncModel(PyFuncModel):

    def __init__(self, model_meta: Model, client: Any, server_pid: int):
        super().__init__(model_meta=model_meta, model_impl=client, predict_fn='invoke')
        self._client = client
        self._server_pid = server_pid

    def predict(self, data, params=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        if inspect.signature(self._client.invoke).parameters.get('params'):
            result = self._client.invoke(data, params=params).get_predictions()
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
            result = self._client.invoke(data).get_predictions()
        if isinstance(result, pandas.DataFrame):
            result = result[result.columns[0]]
        return result

    @property
    def pid(self):
        if self._server_pid is None:
            raise MlflowException('Served PyFunc Model is missing server process ID.')
        return self._server_pid