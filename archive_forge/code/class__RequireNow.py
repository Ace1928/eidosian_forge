import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method
class _RequireNow:
    """Helper callable that accepts all function signatures and simply calls
    :meth:`.LazyDependencyManager.require_now`.  This helpful when used with :func:`.wrap_method`,
    as the callable needs to be compatible with all signatures and be picklable."""
    __slots__ = ('_tester', '_feature')

    def __init__(self, tester, feature):
        self._tester = tester
        self._feature = feature

    def __call__(self, *_args, **_kwargs):
        self._tester.require_now(self._feature)