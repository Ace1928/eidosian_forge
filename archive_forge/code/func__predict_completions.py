import itertools
import logging
import os
import warnings
from string import Formatter
from typing import Any, Dict, Optional, Set
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.openai_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _predict_completions(self, data, params):
    from mlflow.openai.api_request_parallel_processor import process_api_requests
    _validate_model_params(self.task, self.model, params)
    prompts_list = self.format_completions(self.get_params_list(data))
    batch_size = params.pop('batch_size', self.api_config.batch_size)
    _logger.debug(f'Requests are being batched by {batch_size} samples.')
    requests = [{**self.model, **params, 'prompt': prompts_list[i:i + batch_size]} for i in range(0, len(prompts_list), batch_size)]
    request_url = self._construct_request_url('completions', REQUEST_URL_COMPLETIONS)
    results = process_api_requests(requests, request_url, api_token=self.api_token, max_requests_per_minute=self.api_config.max_requests_per_minute, max_tokens_per_minute=self.api_config.max_tokens_per_minute)
    return [row['text'] for batch in results for row in batch['choices']]