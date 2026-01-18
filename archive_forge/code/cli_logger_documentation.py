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
Prompt the user for some text input.

        Args:
            msg: The mesage to display to the user before the prompt.

        Returns:
            The string entered by the user.
        