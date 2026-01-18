from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
class NoOpPathWatcher:

    def __init__(self, _path_str: str, _on_changed: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False):
        pass