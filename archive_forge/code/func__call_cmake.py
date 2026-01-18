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
def _call_cmake(self, args: T.List[str], cmake_file: str, env: T.Optional[T.Dict[str, str]]=None) -> T.Tuple[int, T.Optional[str], T.Optional[str]]:
    build_dir = self._setup_cmake_dir(cmake_file)
    return self.cmakebin.call(args, build_dir, env=env)