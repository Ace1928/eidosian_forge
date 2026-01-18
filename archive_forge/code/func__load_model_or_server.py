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
def _load_model_or_server(model_uri: str, env_manager: str, model_config: Optional[Dict[str, Any]]=None):
    """
    Load a model with env restoration. If a non-local ``env_manager`` is specified, prepare an
    independent Python environment with the training time dependencies of the specified model
    installed and start a MLflow Model Scoring Server process with that model in that environment.
    Return a _ServedPyFuncModel that invokes the scoring server for prediction. Otherwise, load and
    return the model locally as a PyFuncModel using :py:func:`mlflow.pyfunc.load_model`.

    Args:
        model_uri: The uri of the model.
        env_manager: The environment manager to load the model.
        model_config: The model configuration to use by the model, only if the model
                      accepts it.

    Returns:
        A _ServedPyFuncModel for non-local ``env_manager``s or a PyFuncModel otherwise.
    """
    from mlflow.pyfunc.scoring_server.client import ScoringServerClient, StdinScoringServerClient
    if env_manager == _EnvManager.LOCAL:
        return load_model(model_uri, model_config=model_config)
    _logger.info('Starting model server for model environment restoration.')
    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
    is_port_connectable = check_port_connectivity()
    pyfunc_backend = get_flavor_backend(local_path, env_manager=env_manager, install_mlflow=os.environ.get('MLFLOW_HOME') is not None, create_env_root_dir=not is_port_connectable)
    _logger.info('Restoring model environment. This can take a few minutes.')
    pyfunc_backend.prepare_env(model_uri=local_path, capture_output=is_in_databricks_runtime())
    if is_port_connectable:
        server_port = find_free_port()
        scoring_server_proc = pyfunc_backend.serve(model_uri=local_path, port=server_port, host='127.0.0.1', timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(), enable_mlserver=False, synchronous=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        client = ScoringServerClient('127.0.0.1', server_port)
    else:
        scoring_server_proc = pyfunc_backend.serve_stdin(local_path)
        client = StdinScoringServerClient(scoring_server_proc)
    _logger.info(f'Scoring server process started at PID: {scoring_server_proc.pid}')
    try:
        client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
    except Exception as e:
        raise MlflowException('MLflow model server failed to launch.') from e
    return _ServedPyFuncModel(model_meta=model_meta, client=client, server_pid=scoring_server_proc.pid)