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
def _update_config_with_toml(raw_toml: str, where_defined: str) -> None:
    """Update the config system by parsing this string.

    This should only be called from get_config_options.

    Parameters
    ----------
    raw_toml : str
        The TOML file to parse to update the config values.
    where_defined : str
        Tells the config system where this was set.

    """
    import toml
    parsed_config_file = toml.loads(raw_toml)
    for section, options in parsed_config_file.items():
        for name, value in options.items():
            value = _maybe_read_env_variable(value)
            _set_option(f'{section}.{name}', value, where_defined)