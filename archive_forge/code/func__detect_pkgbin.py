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
def _detect_pkgbin(self) -> None:
    for potential_pkgbin in find_external_program(self.env, self.for_machine, 'pkg-config', 'Pkg-config', self.env.default_pkgconfig, allow_default_for_cross=False):
        version_if_ok = self._check_pkgconfig(potential_pkgbin)
        if version_if_ok:
            self.pkgbin = potential_pkgbin
            self.pkgbin_version = version_if_ok
            return
    self.pkgbin = None