from __future__ import annotations
import random
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Final
import streamlit.logger as logger
def _get_installed_streamlit_version() -> Version:
    """Return the streamlit version string from setup.py.

    Returns
    -------
    str
        The version string specified in setup.py.

    """
    return _version_str_to_obj(STREAMLIT_VERSION_STRING)