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
def _validate_cmdfinalization_callable(cls, func: Callable[[plugin.CommandFinalizationData], plugin.CommandFinalizationData]) -> None:
    """Check parameter and return types for command finalization hooks."""
    cls._validate_callable_param_count(func, 1)
    signature = inspect.signature(func)
    _, param = list(signature.parameters.items())[0]
    if param.annotation != plugin.CommandFinalizationData:
        raise TypeError(f'{func.__name__} must have one parameter declared with type {plugin.CommandFinalizationData}')
    if signature.return_annotation != plugin.CommandFinalizationData:
        raise TypeError('{func.__name__} must declare return a return type of {plugin.CommandFinalizationData}')