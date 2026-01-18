import json
import logging
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pydantic
import yaml
from packaging import version
from packaging.version import Version
from pydantic import ConfigDict, Field, ValidationError, root_validator, validator
from pydantic.json import pydantic_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel, LimitModel, ResponseModel
from mlflow.gateway.constants import (
from mlflow.gateway.utils import (
def _resolve_api_key_from_input(api_key_input):
    """
    Resolves the provided API key.

    Input formats accepted:

    - Path to a file as a string which will have the key loaded from it
    - environment variable name that stores the api key
    - the api key itself
    """
    if not isinstance(api_key_input, str):
        raise MlflowException.invalid_parameter_value('The api key provided is not a string. Please provide either an environment variable key, a path to a file containing the api key, or the api key itself')
    if api_key_input.startswith('$'):
        env_var_name = api_key_input[1:]
        if (env_var := os.getenv(env_var_name)):
            return env_var
        else:
            raise MlflowException.invalid_parameter_value(f'Environment variable {env_var_name!r} is not set')
    file = pathlib.Path(api_key_input)
    if file.is_file():
        return file.read_text()
    return api_key_input