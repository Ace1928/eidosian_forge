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
def _setup_env(self, env: EnvironOrDict, uninstalled: bool=False) -> T.Dict[str, str]:
    envvars = self._get_env(uninstalled)
    env = envvars.get_env(env)
    for key, value in env.items():
        if key.startswith('PKG_'):
            mlog.debug(f'env[{key}]: {value}')
    return env