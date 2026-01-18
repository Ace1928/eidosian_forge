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
def _validate_transformers_model_dict(transformers_model):
    """
    Validator for a submitted save dictionary for the transformers model. If any additional keys
    are provided, raise to indicate which invalid keys were submitted.
    """
    if isinstance(transformers_model, dict):
        invalid_keys = [key for key in transformers_model.keys() if key not in _SUPPORTED_SAVE_KEYS]
        if invalid_keys:
            raise MlflowException(f"Invalid dictionary submitted for 'transformers_model'. The key(s) {invalid_keys} are not permitted. Must be one of: {_SUPPORTED_SAVE_KEYS}", error_code=INVALID_PARAMETER_VALUE)
        if FlavorKey.MODEL not in transformers_model:
            raise MlflowException(f"The 'transformers_model' dictionary must have an entry for {FlavorKey.MODEL}", error_code=INVALID_PARAMETER_VALUE)
        model = transformers_model[FlavorKey.MODEL]
    else:
        model = transformers_model.model
    if not hasattr(model, 'name_or_path'):
        raise MlflowException(f"The submitted model type {type(model).__name__} does not inherit from a transformers pre-trained model. It is missing the attribute 'name_or_path'. Please verify that the model is a supported transformers model.", error_code=INVALID_PARAMETER_VALUE)