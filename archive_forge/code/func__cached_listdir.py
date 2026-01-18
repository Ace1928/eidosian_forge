from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import is_windows, MesonException, PerMachine, stringlistify, extract_as_list
from ..cmake import CMakeExecutor, CMakeTraceParser, CMakeException, CMakeToolchain, CMakeExecScope, check_cmake_args, resolve_cmake_trace_targets, cmake_is_debug
from .. import mlog
import importlib.resources
from pathlib import Path
import functools
import re
import os
import shutil
import textwrap
import typing as T
@staticmethod
@functools.lru_cache(maxsize=None)
def _cached_listdir(path: str) -> T.Tuple[T.Tuple[str, str], ...]:
    try:
        return tuple(((x, str(x).lower()) for x in os.listdir(path)))
    except OSError:
        return tuple()