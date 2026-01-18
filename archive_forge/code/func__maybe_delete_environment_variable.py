from __future__ import annotations
import os
import threading
from copy import deepcopy
from typing import (
from blinker import Signal
import streamlit as st
import streamlit.watcher.path_watcher
from streamlit import file_util, runtime
from streamlit.logger import get_logger
@staticmethod
def _maybe_delete_environment_variable(k: Any, v: Any) -> None:
    """Remove the given key/value pair from os.environ if the value
        is a string, int, or float.
        """
    value_type = type(v)
    if value_type in (str, int, float) and os.environ.get(k) == v:
        del os.environ[k]