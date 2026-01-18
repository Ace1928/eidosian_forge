from __future__ import annotations
import contextlib
import inspect
import os
import sys
import threading
import time
import uuid
from collections.abc import Sized
from functools import wraps
from typing import Any, Callable, Final, TypeVar, cast, overload
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Argument, Command
def _get_command_telemetry(_command_func: Callable[..., Any], _command_name: str, *args, **kwargs) -> Command:
    """Get telemetry information for the given callable and its arguments."""
    arg_keywords = inspect.getfullargspec(_command_func).args
    self_arg: Any | None = None
    arguments: list[Argument] = []
    is_method = inspect.ismethod(_command_func)
    name = _command_name
    for i, arg in enumerate(args):
        pos = i
        if is_method:
            i = i + 1
        keyword = arg_keywords[i] if len(arg_keywords) > i else f'{i}'
        if keyword == 'self':
            self_arg = arg
            continue
        argument = Argument(k=keyword, t=_get_type_name(arg), p=pos)
        arg_metadata = _get_arg_metadata(arg)
        if arg_metadata:
            argument.m = arg_metadata
        arguments.append(argument)
    for kwarg, kwarg_value in kwargs.items():
        argument = Argument(k=kwarg, t=_get_type_name(kwarg_value))
        arg_metadata = _get_arg_metadata(kwarg_value)
        if arg_metadata:
            argument.m = arg_metadata
        arguments.append(argument)
    top_level_module = _get_top_level_module(_command_func)
    if top_level_module != 'streamlit':
        name = f'external:{top_level_module}:{name}'
    if name == 'create_instance' and self_arg and hasattr(self_arg, 'name') and self_arg.name:
        name = f'component:{self_arg.name}'
    return Command(name=name, args=arguments)