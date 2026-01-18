from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import PydanticUndefined
from ._config import ConfigWrapper
from ._utils import is_valid_identifier
Return the alias if it is a valid alias and identifier, else None.