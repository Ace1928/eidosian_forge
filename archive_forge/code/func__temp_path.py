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
@contextlib.contextmanager
def _temp_path(dir: tempfile.TemporaryDirectory):
    """
    Wrap tempfile.TemporyDirectory to return a pathlib object.
    """
    with dir as result:
        yield pathlib.Path(result)