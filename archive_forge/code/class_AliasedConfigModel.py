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
class AliasedConfigModel(ConfigModel):
    """
    Enables use of field aliases in a configuration model for backwards compatibility
    """
    if Version(pydantic.__version__) >= Version('2.0'):
        model_config = ConfigDict(populate_by_name=True)
    else:

        class Config:
            allow_population_by_field_name = True