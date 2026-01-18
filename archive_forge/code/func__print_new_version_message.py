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
def _print_new_version_message() -> None:
    if version.should_show_new_version_notice():
        NEW_VERSION_TEXT: Final = "\n  %(new_version)s\n\n  See what's new at https://discuss.streamlit.io/c/announcements\n\n  Enter the following command to upgrade:\n  %(prompt)s %(command)s\n" % {'new_version': cli_util.style_for_cli('A new version of Streamlit is available.', fg='blue', bold=True), 'prompt': cli_util.style_for_cli('$', fg='blue'), 'command': cli_util.style_for_cli('pip install streamlit --upgrade', bold=True)}
        cli_util.print_to_cli(NEW_VERSION_TEXT)