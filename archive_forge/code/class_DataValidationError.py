import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
class DataValidationError(ValueError):

    def __init__(self, name: str, X: Any, Y: Any, errors: Union[Sequence[Mapping[str, Any]], List[Dict[str, Any]]]=[]) -> None:
        """Custom error for validating inputs / outputs at runtime."""
        message = f"Data validation error in '{name}'"
        type_info = f'X: {type(X)} Y: {type(Y)}'
        data = []
        for error in errors:
            err_loc = ' -> '.join([str(p) for p in error.get('loc', [])])
            data.append((err_loc, error.get('msg')))
        result = [message, type_info, table(data)]
        ValueError.__init__(self, '\n\n' + '\n'.join(result))