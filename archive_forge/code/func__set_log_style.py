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
def _set_log_style(self, x):
    """Configures interactivity and formatting."""
    self._log_style = x.lower()
    self.interactive = _isatty()
    if self._log_style == 'auto':
        self.pretty = _isatty()
    elif self._log_style == 'record':
        self.pretty = False
        self._set_color_mode('false')
    elif self._log_style == 'pretty':
        self.pretty = True