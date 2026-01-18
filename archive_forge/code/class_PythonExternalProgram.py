from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
class PythonExternalProgram(BasicPythonExternalProgram):
    run_bytecompile: T.ClassVar[T.Dict[str, bool]] = {}

    def sanity(self, state: T.Optional['ModuleState']=None) -> bool:
        ret = super().sanity()
        if ret:
            self.platlib = self._get_path(state, 'platlib')
            self.purelib = self._get_path(state, 'purelib')
            self.run_bytecompile.setdefault(self.info['version'], False)
        return ret

    def _get_path(self, state: T.Optional['ModuleState'], key: str) -> str:
        rel_path = self.info['install_paths'][key][1:]
        if not state:
            return rel_path
        value = T.cast('str', state.get_option(f'{key}dir', module='python'))
        if value:
            if state.is_user_defined_option('install_env', module='python'):
                raise mesonlib.MesonException(f'python.{key}dir and python.install_env are mutually exclusive')
            return value
        install_env = state.get_option('install_env', module='python')
        if install_env == 'auto':
            install_env = 'venv' if self.info['is_venv'] else 'system'
        if install_env == 'system':
            rel_path = os.path.join(self.info['variables']['prefix'], rel_path)
        elif install_env == 'venv':
            if not self.info['is_venv']:
                raise mesonlib.MesonException('python.install_env cannot be set to "venv" unless you are in a venv!')
            rel_path = self.info['sysconfig_paths'][key]
        return rel_path