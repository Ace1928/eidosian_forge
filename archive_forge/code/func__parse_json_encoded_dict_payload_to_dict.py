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
@staticmethod
def _parse_json_encoded_dict_payload_to_dict(data, key_to_unpack):
    """
        Parses complex dict input types that have been json encoded. Pipelines like
        TableQuestionAnswering require such input types.
        """
    if isinstance(data, list):
        return [{key: json.loads(value) if key == key_to_unpack and isinstance(value, str) else value for key, value in entry.items()} for entry in data]
    elif isinstance(data, dict):
        output = {}
        for key, value in data.items():
            if key == key_to_unpack:
                if isinstance(value, np.ndarray):
                    output[key] = ast.literal_eval(value.item())
                else:
                    output[key] = ast.literal_eval(value)
            elif isinstance(value, np.ndarray):
                output[key] = value.item()
            else:
                output[key] = value
        return output
    else:
        return {key: json.loads(value) if key == key_to_unpack and isinstance(value, str) else value for key, value in data.items()}