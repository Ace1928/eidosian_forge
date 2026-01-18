import inspect
import sys
from copy import copy
from typing import Any, Callable, Dict, List, Tuple, Type, cast, get_type_hints
from typing_extensions import Annotated
from ._typing import get_args, get_origin
from .models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta
class MultipleTyperAnnotationsError(Exception):
    argument_name: str

    def __init__(self, argument_name: str):
        self.argument_name = argument_name

    def __str__(self) -> str:
        return f'Cannot specify multiple `Annotated` Typer arguments for {self.argument_name!r}'