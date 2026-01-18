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
def _derived_from(cls, method, ua_args=None, extra='', skipblocks=0, inconsistencies=None):
    """Helper function for derived_from to ease testing"""
    ua_args = ua_args or []
    original_method = getattr(cls, method.__name__)
    doc = getattr(original_method, '__doc__', None)
    if isinstance(original_method, property):
        original_method = original_method.fget
        if not doc:
            doc = getattr(original_method, '__doc__', None)
    if isinstance(original_method, functools.cached_property):
        original_method = original_method.func
        if not doc:
            doc = getattr(original_method, '__doc__', None)
    if doc is None:
        doc = ''
    if not doc and cls.__name__ in {'DataFrame', 'Series'}:
        for obj in cls.mro():
            obj_method = getattr(obj, method.__name__, None)
            if obj_method is not None and obj_method.__doc__:
                doc = obj_method.__doc__
                break
    if doc:
        doc = ignore_warning(doc, cls, method.__name__, extra=extra, skipblocks=skipblocks, inconsistencies=inconsistencies)
    elif extra:
        doc += extra.rstrip('\n') + '\n\n'
    try:
        method_args = get_named_args(method)
        original_args = get_named_args(original_method)
        not_supported = [m for m in original_args if m not in method_args]
    except ValueError:
        not_supported = []
    if len(ua_args) > 0:
        not_supported.extend(ua_args)
    if len(not_supported) > 0:
        doc = unsupported_arguments(doc, not_supported)
    doc = skip_doctest(doc)
    doc = extra_titles(doc)
    return doc