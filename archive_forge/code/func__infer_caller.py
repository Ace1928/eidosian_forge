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
def _infer_caller():
    """
    Walk the stack and find the frame of the first caller not in this module.
    """

    def is_this_file(frame_info):
        return frame_info.filename == __file__

    def is_wrapper(frame_info):
        return frame_info.function == 'wrapper'
    not_this_file = itertools.filterfalse(is_this_file, inspect.stack())
    callers = itertools.filterfalse(is_wrapper, not_this_file)
    return next(callers).frame