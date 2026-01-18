from __future__ import annotations
import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
from mlflow import pyfunc
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
from mlflow.transformers.hub_utils import is_valid_hf_repo_id
from mlflow.transformers.llm_inference_utils import (
from mlflow.transformers.model_io import (
from mlflow.transformers.peft import (
from mlflow.transformers.signature import (
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _strip_input_from_response_in_instruction_pipelines(self, input_data, output, output_key, flavor_config, include_prompt=True, collapse_whitespace=False):
    """
        Parse the output from instruction pipelines to conform with other text generator
        pipeline types and remove line feed characters and other confusing outputs
        """

    def extract_response_data(data_out):
        if all((isinstance(x, dict) for x in data_out)):
            return [elem[output_key] for elem in data_out][0]
        elif all((isinstance(x, list) for x in data_out)):
            return [elem[output_key] for coll in data_out for elem in coll]
        else:
            raise MlflowException(f'Unable to parse the pipeline output. Expected List[Dict[str,str]] or List[List[Dict[str,str]]] but got {type(data_out)} instead.')
    output = extract_response_data(output)

    def trim_input(data_in, data_out):
        if not include_prompt and flavor_config[FlavorKey.INSTANCE_TYPE] in self._supported_custom_generator_types and data_out.startswith(data_in + '\n\n'):
            data_out = data_out[len(data_in):].lstrip()
            if data_out.startswith('A:'):
                data_out = data_out[2:].lstrip()
        if collapse_whitespace:
            data_out = re.sub('\\s+', ' ', data_out).strip()
        return data_out
    if isinstance(input_data, list) and isinstance(output, list):
        return [trim_input(data_in, data_out) for data_in, data_out in zip(input_data, output)]
    elif isinstance(input_data, str) and isinstance(output, str):
        return trim_input(input_data, output)
    else:
        raise MlflowException(f'Unknown data structure after parsing output. Expected str or List[str]. Got {type(output)} instead.')