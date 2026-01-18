import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
@classmethod
def _validate_prepostcmd_hook(cls, func: Callable[[CommandDataType], CommandDataType], data_type: Type[CommandDataType]) -> None:
    """Check parameter and return types for pre and post command hooks."""
    signature = inspect.signature(func)
    cls._validate_callable_param_count(cast(Callable[..., Any], func), 1)
    paramname = list(signature.parameters.keys())[0]
    param = signature.parameters[paramname]
    if param.annotation != data_type:
        raise TypeError(f'argument 1 of {func.__name__} has incompatible type {param.annotation}, expected {data_type}')
    if signature.return_annotation == signature.empty:
        raise TypeError(f'{func.__name__} does not have a declared return type, expected {data_type}')
    if signature.return_annotation != data_type:
        raise TypeError(f'{func.__name__} has incompatible return type {signature.return_annotation}, expected {data_type}')