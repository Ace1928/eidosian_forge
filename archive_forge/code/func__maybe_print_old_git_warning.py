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
def _maybe_print_old_git_warning(main_script_path: str) -> None:
    """If our script is running in a Git repo, and we're running a very old
    Git version, print a warning that Git integration will be unavailable.
    """
    repo = GitRepo(main_script_path)
    if not repo.is_valid() and repo.git_version is not None and (repo.git_version < MIN_GIT_VERSION):
        git_version_string = '.'.join((str(val) for val in repo.git_version))
        min_version_string = '.'.join((str(val) for val in MIN_GIT_VERSION))
        cli_util.print_to_cli('')
        cli_util.print_to_cli('  Git integration is disabled.', fg='yellow', bold=True)
        cli_util.print_to_cli('')
        cli_util.print_to_cli(f'  Streamlit requires Git {min_version_string} or later, but you have {git_version_string}.', fg='yellow')
        cli_util.print_to_cli('  Git is used by Streamlit Cloud (https://streamlit.io/cloud).', fg='yellow')
        cli_util.print_to_cli('  To enable this feature, please update Git.', fg='yellow')