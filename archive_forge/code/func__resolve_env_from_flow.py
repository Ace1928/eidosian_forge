import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _resolve_env_from_flow(flow_dag_path):
    with open(flow_dag_path) as f:
        flow_dict = yaml.safe_load(f)
    environment = flow_dict.get('environment', {})
    if _FLOW_ENV_REQUIREMENTS in environment:
        environment[_FLOW_ENV_REQUIREMENTS] = f'{_MODEL_FLOW_DIRECTORY}/{environment[_FLOW_ENV_REQUIREMENTS]}'
    return environment