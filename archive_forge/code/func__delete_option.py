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
def _delete_option(key: str) -> None:
    """Remove a ConfigOption by key from the global store.

    Only for use in testing.
    """
    try:
        del _config_options_template[key]
        assert _config_options is not None, '_config_options should always be populated here.'
        del _config_options[key]
    except Exception:
        pass