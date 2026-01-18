import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def _patched_makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
    """Monkey-patched version of logging.Logger.makeRecord
    We have to patch default loggers so they use the proper frame for
    line numbers and function names (otherwise everything shows up as
    e.g. cli_logger:info() instead of as where it was called from).

    In Python 3.8 we could just use stacklevel=2, but we have to support
    Python 3.6 and 3.7 as well.

    The solution is this Python magic superhack.

    The default makeRecord will deliberately check that we don't override
    any existing property on the LogRecord using `extra`,
    so we remove that check.

    This patched version is otherwise identical to the one in the standard
    library.

    TODO: Remove this magic superhack. Find a more responsible workaround.
    """
    rv = logging._logRecordFactory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    if extra is not None:
        rv.__dict__.update(extra)
    return rv