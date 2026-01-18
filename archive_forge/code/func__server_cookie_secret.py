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
@_create_option('server.cookieSecret', type_=str, sensitive=True)
@util.memoize
def _server_cookie_secret() -> str:
    """Symmetric key used to produce signed cookies. If deploying on multiple replicas, this should
    be set to the same value across all replicas to ensure they all share the same secret.

    Default: randomly generated secret key.
    """
    return secrets.token_hex()