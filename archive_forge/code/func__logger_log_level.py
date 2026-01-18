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
@_create_option('logger.level', type_=str)
def _logger_log_level() -> str:
    """Level of logging: 'error', 'warning', 'info', or 'debug'.

    Default: 'info'
    """
    if get_option('global.logLevel'):
        return str(get_option('global.logLevel'))
    elif get_option('global.developmentMode'):
        return 'debug'
    else:
        return 'info'