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
def _coerce_exploded_dict_to_single_dict(self, data):
    """
        Parses the result of Pandas DataFrame.to_dict(orient="records") from pyfunc
        signature validation to coerce the output to the required format for a
        Pipeline that requires a single dict with list elements such as
        TableQuestionAnsweringPipeline.
        Example input:

        [
          {"answer": "We should order more pizzas to meet the demand."},
          {"answer": "The venue size should be updated to handle the number of guests."},
        ]

        Output:

        [
          "We should order more pizzas to meet the demand.",
          "The venue size should be updated to handle the number of guests.",
        ]

        """
    import transformers
    if not isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
        return data
    elif isinstance(data, list) and all((isinstance(item, dict) for item in data)):
        collection = data.copy()
        parsed = collection[0]
        for coll in collection:
            for key, value in coll.items():
                if key not in parsed:
                    raise MlflowException('Unable to parse the input. The keys within each dictionary of the parsed input are not consistentamong the dictionaries.', error_code=INVALID_PARAMETER_VALUE)
                if value != parsed[key]:
                    value_type = type(parsed[key])
                    if value_type == str:
                        parsed[key] = [parsed[key], value]
                    elif value_type == list:
                        if all((len(entry) == 1 for entry in value)):
                            parsed[key].append([str(value)][0])
                        else:
                            parsed[key] = parsed[key].append(value)
                    else:
                        parsed[key] = value
        return parsed
    else:
        return data