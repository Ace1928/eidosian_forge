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
def _maybe_print_static_folder_warning(main_script_path: str) -> None:
    """Prints a warning if the static folder is misconfigured."""
    if config.get_option('server.enableStaticServing'):
        static_folder_path = file_util.get_app_static_dir(main_script_path)
        if not os.path.isdir(static_folder_path):
            cli_util.print_to_cli(f'WARNING: Static file serving is enabled, but no static folder found at {static_folder_path}. To disable static file serving, set server.enableStaticServing to false.', fg='yellow')
        else:
            static_folder_size = file_util.get_directory_size(static_folder_path)
            if static_folder_size > MAX_APP_STATIC_FOLDER_SIZE:
                config.set_option('server.enableStaticServing', False)
                cli_util.print_to_cli('WARNING: Static folder size is larger than 1GB. Static file serving has been disabled.', fg='yellow')