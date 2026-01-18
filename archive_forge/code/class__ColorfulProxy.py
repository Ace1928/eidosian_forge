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
class _ColorfulProxy:
    _proxy_allowlist = ['disable', 'reset', 'bold', 'italic', 'underlined', 'dimmed', 'dodgerBlue', 'limeGreen', 'red', 'orange', 'skyBlue', 'magenta', 'yellow']

    def __getattr__(self, name):
        res = getattr(_cf, name)
        if callable(res) and name not in _ColorfulProxy._proxy_allowlist:
            raise ValueError("Usage of the colorful method '" + name + "' is forbidden by the proxy to keep a consistent color scheme. Check `cli_logger.py` for allowed methods")
        return res