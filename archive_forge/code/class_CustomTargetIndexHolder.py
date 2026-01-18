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
class CustomTargetIndexHolder(ObjectHolder[build.CustomTargetIndex]):

    def __init__(self, target: build.CustomTargetIndex, interp: 'Interpreter'):
        super().__init__(target, interp)
        self.methods.update({'full_path': self.full_path_method})

    @FeatureNew('custom_target[i].full_path', '0.54.0')
    @noPosargs
    @noKwargs
    def full_path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        assert self.interpreter.backend is not None
        return self.interpreter.backend.get_target_filename_abs(self.held_object)