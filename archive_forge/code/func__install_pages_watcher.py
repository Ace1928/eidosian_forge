from __future__ import annotations
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Final
from streamlit import (
from streamlit.config import CONFIG_FILENAMES
from streamlit.git_util import MIN_GIT_VERSION, GitRepo
from streamlit.logger import get_logger
from streamlit.source_util import invalidate_pages_cache
from streamlit.watcher import report_watchdog_availability, watch_dir, watch_file
from streamlit.web.server import Server, server_address_is_unix_socket, server_util
def _install_pages_watcher(main_script_path_str: str) -> None:

    def _on_pages_changed(_path: str) -> None:
        invalidate_pages_cache()
    main_script_path = Path(main_script_path_str)
    pages_dir = main_script_path.parent / 'pages'
    watch_dir(str(pages_dir), _on_pages_changed, glob_pattern='*.py', allow_nonexistent=True)