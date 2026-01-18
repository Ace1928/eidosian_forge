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
class SilentClickException(click.ClickException):
    """`ClickException` that does not print a message.

    Some of our tooling relies on catching ClickException in particular.

    However the default prints a message, which is undesirable since we expect
    our code to log errors manually using `cli_logger.error()` to allow for
    colors and other formatting.
    """

    def __init__(self, message: str):
        super(SilentClickException, self).__init__(message)

    def show(self, file=None):
        pass