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
@_create_option('server.headless', type_=bool)
def _server_headless() -> bool:
    """If false, will attempt to open a browser window on start.

    Default: false unless (1) we are on a Linux box where DISPLAY is unset, or
    (2) we are running in the Streamlit Atom plugin.
    """
    if env_util.IS_LINUX_OR_BSD and (not os.getenv('DISPLAY')):
        return True
    if os.getenv('IS_RUNNING_IN_STREAMLIT_EDITOR_PLUGIN') is not None:
        return True
    return False