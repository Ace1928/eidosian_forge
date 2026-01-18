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
@_create_option('global.developmentMode', visibility='hidden', type_=bool)
def _global_development_mode() -> bool:
    """Are we in development mode.

    This option defaults to True if and only if Streamlit wasn't installed
    normally.
    """
    return not env_util.is_pex() and 'site-packages' not in __file__ and ('dist-packages' not in __file__) and ('__pypackages__' not in __file__)