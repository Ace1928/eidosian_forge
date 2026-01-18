from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def _is_watchdog_available() -> bool:
    """Check if the watchdog module is installed."""
    try:
        import watchdog
        return True
    except ImportError:
        return False