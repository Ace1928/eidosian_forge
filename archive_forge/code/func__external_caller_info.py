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
def _external_caller_info():
    """Get the info from the caller frame.

    Used to override the logging function and line number with the correct
    ones. See the comment on _patched_makeRecord for more info.
    """
    frame = inspect.currentframe()
    caller = frame
    levels = 0
    while caller.f_code.co_filename == __file__:
        caller = caller.f_back
        levels += 1
    return {'lineno': caller.f_lineno, 'filename': os.path.basename(caller.f_code.co_filename)}