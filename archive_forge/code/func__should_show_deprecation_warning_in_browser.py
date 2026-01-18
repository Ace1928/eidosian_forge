from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def _should_show_deprecation_warning_in_browser() -> bool:
    """True if we should print deprecation warnings to the browser."""
    return bool(config.get_option('client.showErrorDetails'))