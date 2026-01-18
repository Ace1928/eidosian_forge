from __future__ import annotations
import ast
import contextlib
import inspect
import re
import types
from typing import TYPE_CHECKING, Any, Final, cast
import streamlit
from streamlit.proto.DocString_pb2 import DocString as DocStringProto
from streamlit.proto.DocString_pb2 import Member as MemberProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_runner import (
from streamlit.runtime.secrets import Secrets
from streamlit.string_util import is_mem_address_str
def _get_human_readable_value(value):
    if isinstance(value, Secrets):
        return None
    if inspect.isclass(value) or inspect.ismodule(value) or callable(value):
        return None
    value_str = repr(value)
    if isinstance(value, str):
        return _shorten(value_str)
    if is_mem_address_str(value_str):
        return None
    return _shorten(value_str)