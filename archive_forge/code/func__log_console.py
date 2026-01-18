from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def _log_console(driver: WebDriver) -> None:
    levels = {'WARNING', 'ERROR', 'SEVERE'}
    try:
        logs = driver.get_log('browser')
    except Exception:
        return
    messages = [log.get('message') for log in logs if log.get('level') in levels]
    if len(messages) > 0:
        log.warning('There were browser warnings and/or errors that may have affected your export')
        for message in messages:
            log.warning(message)