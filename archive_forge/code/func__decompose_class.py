import warnings
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType, prepare_class, resolve_bases
from typing import (
from typing_extensions import dataclass_transform
from .class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
from .config import BaseConfig, Extra, inherit_config, prepare_config
from .error_wrappers import ErrorWrapper, ValidationError
from .errors import ConfigError, DictError, ExtraError, MissingError
from .fields import (
from .json import custom_pydantic_encoder, pydantic_encoder
from .parse import Protocol, load_file, load_str_bytes
from .schema import default_ref_template, model_schema
from .types import PyObject, StrBytes
from .typing import (
from .utils import (
@classmethod
def _decompose_class(cls: Type['Model'], obj: Any) -> GetterDict:
    if isinstance(obj, GetterDict):
        return obj
    return cls.__config__.getter_dict(obj)