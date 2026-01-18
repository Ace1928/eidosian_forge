import contextlib
import functools
import importlib.util
import logging
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union
import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.langchain._langchain_autolog import (
from mlflow.langchain._rag_utils import _CODE_CONFIG, _CODE_PATH, _set_config_path
from mlflow.langchain.databricks_dependencies import (
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _load_model_from_local_fs(local_model_path):
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    if _CODE_CONFIG in flavor_conf:
        path = flavor_conf.get(_CODE_CONFIG)
        flavor_code_config = flavor_conf.get(FLAVOR_CONFIG_CODE)
        if path is not None:
            config_path = os.path.join(local_model_path, flavor_code_config, os.path.basename(path))
        else:
            config_path = None
        flavor_code_path = flavor_conf.get(_CODE_PATH, 'chain.py')
        code_path = os.path.join(local_model_path, flavor_code_config, os.path.basename(flavor_code_path))
        return _load_model_code_path(code_path, config_path)
    else:
        _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
        with patch_langchain_type_to_cls_dict():
            return _load_model(local_model_path, flavor_conf)