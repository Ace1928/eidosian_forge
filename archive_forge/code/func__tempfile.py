import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib
from typing import Union, Optional
from .abc import ResourceReader, Traversable
from ._adapters import wrap_spec
@contextlib.contextmanager
def _tempfile(reader, suffix='', *, _os_remove=os.remove):
    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    try:
        try:
            os.write(fd, reader())
        finally:
            os.close(fd)
        del reader
        yield pathlib.Path(raw_path)
    finally:
        try:
            _os_remove(raw_path)
        except FileNotFoundError:
            pass