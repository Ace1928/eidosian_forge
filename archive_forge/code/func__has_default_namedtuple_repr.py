import builtins
import collections
import dataclasses
import inspect
import os
import sys
from array import array
from collections import Counter, UserDict, UserList, defaultdict, deque
from dataclasses import dataclass, fields, is_dataclass
from inspect import isclass
from itertools import islice
from types import MappingProxyType
from typing import (
from pip._vendor.rich.repr import RichReprResult
from . import get_console
from ._loop import loop_last
from ._pick import pick_bool
from .abc import RichRenderable
from .cells import cell_len
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin, JupyterRenderable
from .measure import Measurement
from .text import Text
def _has_default_namedtuple_repr(obj: object) -> bool:
    """Check if an instance of namedtuple contains the default repr

    Args:
        obj (object): A namedtuple

    Returns:
        bool: True if the default repr is used, False if there's a custom repr.
    """
    obj_file = None
    try:
        obj_file = inspect.getfile(obj.__repr__)
    except (OSError, TypeError):
        pass
    default_repr_file = inspect.getfile(_dummy_namedtuple.__repr__)
    return obj_file == default_repr_file