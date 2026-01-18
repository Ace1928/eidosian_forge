from __future__ import annotations
import copy
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Callable
from blinker import Signal
from streamlit import config_util, development, env_util, file_util, util
from streamlit.config_option import ConfigOption
from streamlit.errors import StreamlitAPIException
def _maybe_convert_to_number(v: Any) -> Any:
    """Convert v to int or float, or leave it as is."""
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        pass
    return v