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
@staticmethod
def _define_variable_args(define_variable: PkgConfigDefineType) -> T.List[str]:
    ret = []
    if define_variable:
        for pair in define_variable:
            ret.append('--define-variable=' + '='.join(pair))
    return ret