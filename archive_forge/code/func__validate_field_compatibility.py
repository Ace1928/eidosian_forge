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
@classmethod
def _validate_field_compatibility(cls, info: Dict[str, Any]):
    if not isinstance(info, dict):
        return info
    api_type = (info.get('openai_api_type') or OpenAIAPIType.OPENAI).lower()
    if api_type == OpenAIAPIType.OPENAI:
        if info.get('openai_deployment_name') is not None:
            raise MlflowException.invalid_parameter_value(f"OpenAI route configuration can only specify a value for 'openai_deployment_name' if 'openai_api_type' is '{OpenAIAPIType.AZURE}' or '{OpenAIAPIType.AZUREAD}'. Found type: '{api_type}'")
        if info.get('openai_api_base') is None:
            info['openai_api_base'] = 'https://api.openai.com/v1'
    elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
        if info.get('openai_organization') is not None:
            raise MlflowException.invalid_parameter_value(f"OpenAI route configuration can only specify a value for 'openai_organization' if 'openai_api_type' is '{OpenAIAPIType.OPENAI}'")
        base_url = info.get('openai_api_base')
        deployment_name = info.get('openai_deployment_name')
        api_version = info.get('openai_api_version')
        if (base_url, deployment_name, api_version).count(None) > 0:
            raise MlflowException.invalid_parameter_value(f"OpenAI route configuration must specify 'openai_api_base', 'openai_deployment_name', and 'openai_api_version' if 'openai_api_type' is '{OpenAIAPIType.AZURE}' or '{OpenAIAPIType.AZUREAD}'.")
    else:
        raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{api_type}'")
    return info