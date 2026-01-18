from __future__ import annotations
import codecs
import functools
import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timedelta
from errno import ENOENT
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral, Number
from operator import add
from threading import Lock
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast, overload
from weakref import WeakValueDictionary
import tlz as toolz
from dask import config
from dask.core import get_deps
from dask.typing import no_default
class _HashIdWrapper:
    """Hash and compare a wrapped object by identity instead of value"""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __eq__(self, other):
        if not isinstance(other, _HashIdWrapper):
            return NotImplemented
        return self.wrapped is other.wrapped

    def __ne__(self, other):
        if not isinstance(other, _HashIdWrapper):
            return NotImplemented
        return self.wrapped is not other.wrapped

    def __hash__(self):
        return id(self.wrapped)