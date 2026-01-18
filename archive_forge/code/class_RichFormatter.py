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
class RichFormatter(BaseFormatter):
    pprint: bool = True

    def __call__(self, value: Any) -> Any:
        if self.pprint:
            return _ipy_display_hook(value, console=get_console(), overflow=overflow, indent_guides=indent_guides, max_length=max_length, max_string=max_string, max_depth=max_depth, expand_all=expand_all)
        else:
            return repr(value)