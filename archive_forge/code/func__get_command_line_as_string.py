from __future__ import annotations
import os
import sys
from typing import Any
import click
import streamlit.runtime.caching as caching
import streamlit.runtime.legacy_caching as legacy_caching
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config
from streamlit.config_option import ConfigOption
from streamlit.runtime.credentials import Credentials, check_credentials
from streamlit.web.cache_storage_manager_config import (
def _get_command_line_as_string() -> str | None:
    import subprocess
    parent = click.get_current_context().parent
    if parent is None:
        return None
    if 'streamlit.cli' in parent.command_path:
        raise RuntimeError('Running streamlit via `python -m streamlit.cli <command>` is unsupported. Please use `python -m streamlit <command>` instead.')
    cmd_line_as_list = [parent.command_path]
    cmd_line_as_list.extend(sys.argv[1:])
    return subprocess.list2cmdline(cmd_line_as_list)