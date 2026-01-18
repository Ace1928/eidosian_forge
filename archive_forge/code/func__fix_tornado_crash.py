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
def _fix_tornado_crash() -> None:
    """Set default asyncio policy to be compatible with Tornado 6.

    Tornado 6 (at least) is not compatible with the default
    asyncio implementation on Windows. So here we
    pick the older SelectorEventLoopPolicy when the OS is Windows
    if the known-incompatible default policy is in use.

    This has to happen as early as possible to make it a low priority and
    overridable

    See: https://github.com/tornadoweb/tornado/issues/2608

    FIXME: if/when tornado supports the defaults in asyncio,
    remove and bump tornado requirement for py38
    """
    if env_util.IS_WINDOWS:
        try:
            from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
        except ImportError:
            pass
        else:
            if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())