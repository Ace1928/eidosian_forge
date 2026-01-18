from __future__ import annotations
from pathlib import Path
from .base import ExternalDependency, DependencyException, sort_libpaths, DependencyTypeName
from ..mesonlib import EnvironmentVariables, OptionKey, OrderedSet, PerMachine, Popen_safe, Popen_safe_logged, MachineChoice, join_args
from ..programs import find_external_program, ExternalProgram
from .. import mlog
from pathlib import PurePath
from functools import lru_cache
import re
import os
import shlex
import typing as T
def _set_libs(self) -> None:
    libs = self.pkgconfig.libs(self.name, self.static, allow_system=True)
    raw_libs = self.pkgconfig.libs(self.name, self.static, allow_system=False)
    self.link_args, self.raw_link_args = self._search_libs(libs, raw_libs)