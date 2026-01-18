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
def _get_arg_metadata(arg: object) -> str | None:
    """Get metadata information related to the value of the given object."""
    with contextlib.suppress(Exception):
        if isinstance(arg, bool):
            return f'val:{arg}'
        if isinstance(arg, Sized):
            return f'len:{len(arg)}'
    return None