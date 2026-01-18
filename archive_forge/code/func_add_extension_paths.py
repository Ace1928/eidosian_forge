from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
def add_extension_paths(self, paths: T.Union[T.List[str], T.Set[str]]) -> None:
    for path in paths:
        if path in self._extra_extension_paths:
            continue
        self._extra_extension_paths.add(path)
        self.cmd.extend(['--extra-extension-path', path])