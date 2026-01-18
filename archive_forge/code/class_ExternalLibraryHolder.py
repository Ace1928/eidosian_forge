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
class ExternalLibraryHolder(ObjectHolder[ExternalLibrary]):

    def __init__(self, el: ExternalLibrary, interpreter: 'Interpreter'):
        super().__init__(el, interpreter)
        self.methods.update({'found': self.found_method, 'type_name': self.type_name_method, 'partial_dependency': self.partial_dependency_method})

    @noPosargs
    @noKwargs
    def type_name_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.held_object.type_name

    @noPosargs
    @noKwargs
    def found_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.held_object.found()

    @FeatureNew('dependency.partial_dependency', '0.46.0')
    @noPosargs
    @typed_kwargs('dependency.partial_dependency', *_PARTIAL_DEP_KWARGS)
    def partial_dependency_method(self, args: T.List[TYPE_nvar], kwargs: 'kwargs.DependencyMethodPartialDependency') -> Dependency:
        pdep = self.held_object.get_partial_dependency(**kwargs)
        return pdep