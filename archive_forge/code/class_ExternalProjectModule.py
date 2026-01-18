from __future__ import annotations
from pathlib import Path
import os
import shlex
import subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, NewExtensionModule, ModuleInfo
from .. import mlog, build
from ..compilers.compilers import CFLAGS_MAPPING
from ..envconfig import ENV_VAR_PROG_MAP
from ..dependencies import InternalDependency
from ..dependencies.pkgconfig import PkgConfigInterface
from ..interpreterbase import FeatureNew
from ..interpreter.type_checking import ENV_KW, DEPENDS_KW
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import (EnvironmentException, MesonException, Popen_safe, MachineChoice,
class ExternalProjectModule(ExtensionModule):
    INFO = ModuleInfo('External build system', '0.56.0', unstable=True)

    def __init__(self, interpreter: 'Interpreter'):
        super().__init__(interpreter)
        self.methods.update({'add_project': self.add_project})

    @typed_pos_args('external_project_mod.add_project', str)
    @typed_kwargs('external_project.add_project', KwargInfo('configure_options', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('cross_configure_options', ContainerTypeInfo(list, str), default=['--host=@HOST@'], listify=True), KwargInfo('verbose', bool, default=False), ENV_KW, DEPENDS_KW.evolve(since='0.63.0'))
    def add_project(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'AddProject') -> ModuleReturnValue:
        configure_command = args[0]
        project = ExternalProject(state, configure_command, kwargs['configure_options'], kwargs['cross_configure_options'], kwargs['env'], kwargs['verbose'], kwargs['depends'])
        return ModuleReturnValue(project, project.targets)