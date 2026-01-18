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
def _validate_str_input_uri_or_file(input_str):
    """
        Validation of blob references to either audio or image files,
        if a string is input to the ``predict``
        method, perform validation of the string contents by checking for a valid uri or
        filesystem reference instead of surfacing the cryptic stack trace that is otherwise raised
        for an invalid uri input.
        """

    def is_uri(s):
        try:
            result = urlparse(s)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    valid_uri = os.path.isfile(input_str) or is_uri(input_str)
    if not valid_uri:
        if len(input_str) <= 20:
            data_str = f'Received: {input_str}'
        else:
            data_str = f'Received (truncated): {input_str[:20]}...'
        raise MlflowException(f'An invalid string input was provided. String inputs to audio or image files must be either a file location or a uri.audio files must be either a file location or a uri. {data_str}', error_code=BAD_REQUEST)