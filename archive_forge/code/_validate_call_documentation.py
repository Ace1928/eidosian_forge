from __future__ import annotations as _annotations
import inspect
from functools import partial
from typing import Any, Awaitable, Callable
import pydantic_core
from ..config import ConfigDict
from ..plugin._schema_validator import create_schema_validator
from . import _generate_schema, _typing_extra
from ._config import ConfigWrapper
This is a wrapper around a function that validates the arguments passed to it, and optionally the return value.