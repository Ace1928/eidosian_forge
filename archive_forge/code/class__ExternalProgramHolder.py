from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
class _ExternalProgramHolder(ObjectHolder[_EXTPROG]):

    def __init__(self, ep: _EXTPROG, interpreter: 'Interpreter') -> None:
        super().__init__(ep, interpreter)
        self.methods.update({'found': self.found_method, 'path': self.path_method, 'version': self.version_method, 'full_path': self.full_path_method})

    @noPosargs
    @noKwargs
    def found_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.found()

    @noPosargs
    @noKwargs
    @FeatureDeprecated('ExternalProgram.path', '0.55.0', 'use ExternalProgram.full_path() instead')
    def path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self._full_path()

    @noPosargs
    @noKwargs
    @FeatureNew('ExternalProgram.full_path', '0.55.0')
    def full_path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self._full_path()

    def _full_path(self) -> str:
        if not self.found():
            raise InterpreterException('Unable to get the path of a not-found external program')
        path = self.held_object.get_path()
        assert path is not None
        return path

    @noPosargs
    @noKwargs
    @FeatureNew('ExternalProgram.version', '0.62.0')
    def version_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        if not self.found():
            raise InterpreterException('Unable to get the version of a not-found external program')
        try:
            return self.held_object.get_version(self.interpreter)
        except mesonlib.MesonException:
            return 'unknown'

    def found(self) -> bool:
        return self.held_object.found()