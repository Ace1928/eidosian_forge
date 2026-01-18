import inspect
import sys
from copy import copy
from typing import Any, Callable, Dict, List, Tuple, Type, cast, get_type_hints
from typing_extensions import Annotated
from ._typing import get_args, get_origin
from .models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta
class AnnotatedParamWithDefaultValueError(Exception):
    argument_name: str
    param_type: Type[ParameterInfo]

    def __init__(self, argument_name: str, param_type: Type[ParameterInfo]):
        self.argument_name = argument_name
        self.param_type = param_type

    def __str__(self) -> str:
        param_type_str = _param_type_to_user_string(self.param_type)
        return f'{param_type_str} default value cannot be set in `Annotated` for {self.argument_name!r}. Set the default value with `=` instead.'