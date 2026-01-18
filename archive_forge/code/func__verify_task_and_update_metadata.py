import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.llm import (
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
@experimental
def _verify_task_and_update_metadata(task: str, metadata: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if task not in [_LLM_INFERENCE_TASK_EMBEDDING]:
        raise MlflowException.invalid_parameter_value(f'Received invalid parameter value for `task` argument {task}. Task type could only be {_LLM_INFERENCE_TASK_EMBEDDING}')
    if metadata is None:
        metadata = {}
    if 'task' in metadata and metadata['task'] != task:
        raise MlflowException.invalid_parameter_value(f'Received invalid parameter value for `task` argument {task}. Task type is inconsistent with the task value from metadata {metadata['task']}')
    metadata['task'] = task
    return metadata