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
def _get_build_dir(self) -> Path:
    build_dir = Path(self.cmake_root_dir) / f'cmake_{self.name}'
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir