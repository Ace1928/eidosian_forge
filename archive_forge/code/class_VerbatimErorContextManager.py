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
class VerbatimErorContextManager:

    def __enter__(self):
        cli_logger.error(cf.bold('!!! ') + '{}', msg, *args, **kwargs)

    def __exit__(self, type, value, tb):
        cli_logger.error(cf.bold('!!!'))