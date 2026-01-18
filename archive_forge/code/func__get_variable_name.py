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
def _get_variable_name():
    """Try to get the name of the variable in the current line, as set by the user.

    For example:
    foo = bar.Baz(123)
    st.help(foo)

    The name is "foo"
    """
    code = _get_current_line_of_code_as_str()
    if code is None:
        return None
    return _get_variable_name_from_code_str(code)