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
def _is_unset(option_name: str) -> bool:
    """Check if a given option has not been set by the user.

    Parameters
    ----------
    option_name : str
        The option to check


    Returns
    -------
    bool
        True if the option has not been set by the user.

    """
    return get_where_defined(option_name) == ConfigOption.DEFAULT_DEFINITION