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
def _parse_lists_of_dict_to_list_of_str(self, output_data, target_dict_key) -> List[str]:
    """
        Parses the output results from select Pipeline types to extract specific values from a
        target key.
        Examples (with "a" as the `target_dict_key`):

        Input: [{"a": "valid", "b": "invalid"}, {"a": "another valid", "c": invalid"}]
        Output: ["valid", "another_valid"]

        Input: [{"a": "valid", "b": [{"a": "another valid"}, {"b": "invalid"}]},
                {"a": "valid 2", "b": [{"a": "another valid 2"}, {"c": "invalid"}]}]
        Output: ["valid", "another valid", "valid 2", "another valid 2"]
        """
    if isinstance(output_data, list):
        output_coll = []
        for output in output_data:
            if isinstance(output, dict):
                for key, value in output.items():
                    if key == target_dict_key:
                        output_coll.append(output[target_dict_key])
                    elif isinstance(value, list) and all((isinstance(elem, dict) for elem in value)):
                        output_coll.extend(self._parse_lists_of_dict_to_list_of_str(value, target_dict_key))
            elif isinstance(output, list):
                output_coll.extend(self._parse_lists_of_dict_to_list_of_str(output, target_dict_key))
        return output_coll
    elif target_dict_key:
        return output_data[target_dict_key]
    else:
        return output_data