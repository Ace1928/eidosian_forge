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
def _get_top_level_module(func: Callable[..., Any]) -> str:
    """Get the top level module for the given function."""
    module = inspect.getmodule(func)
    if module is None or not module.__name__:
        return 'unknown'
    return module.__name__.split('.')[0]