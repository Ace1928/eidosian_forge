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
def _temp_file(path):
    return _tempfile(path.read_bytes, suffix=path.name)