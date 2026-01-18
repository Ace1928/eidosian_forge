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
def _get_stcommand_arg(tree):
    """Gets the argument node for the st command in tree (AST)."""
    root_node = tree.body[0].value
    if root_node.args:
        return root_node.args[0]
    return None