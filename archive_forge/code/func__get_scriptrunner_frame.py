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
def _get_scriptrunner_frame():
    prev_frame = None
    scriptrunner_frame = None
    for frame in inspect.stack():
        if frame.code_context is None:
            return None
        if frame.filename == SCRIPTRUNNER_FILENAME:
            scriptrunner_frame = prev_frame
            break
        prev_frame = frame
    return scriptrunner_frame