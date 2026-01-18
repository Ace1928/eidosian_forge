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
def _load_route_config(path: Union[str, Path]) -> GatewayConfig:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration RouteConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    try:
        configuration = yaml.safe_load(path.read_text())
    except Exception as e:
        raise MlflowException.invalid_parameter_value(f'The file at {path} is not a valid yaml file') from e
    check_configuration_deprecated_fields(configuration)
    check_configuration_route_name_collisions(configuration)
    try:
        return GatewayConfig(**configuration)
    except ValidationError as e:
        raise MlflowException.invalid_parameter_value(f'The gateway configuration is invalid: {e}') from e