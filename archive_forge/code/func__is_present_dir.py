import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib
import inspect
import warnings
import itertools
from typing import Union, Optional, cast
from .abc import ResourceReader, Traversable
from ._compat import wrap_spec
def _is_present_dir(path: Traversable) -> bool:
    """
    Some Traversables implement ``is_dir()`` to raise an
    exception (i.e. ``FileNotFoundError``) when the
    directory doesn't exist. This function wraps that call
    to always return a boolean and only return True
    if there's a dir and it exists.
    """
    with contextlib.suppress(FileNotFoundError):
        return path.is_dir()
    return False