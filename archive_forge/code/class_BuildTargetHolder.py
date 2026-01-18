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
class BuildTargetHolder(ObjectHolder[_BuildTarget]):

    def __init__(self, target: _BuildTarget, interp: 'Interpreter'):
        super().__init__(target, interp)
        self.methods.update({'extract_objects': self.extract_objects_method, 'extract_all_objects': self.extract_all_objects_method, 'name': self.name_method, 'get_id': self.get_id_method, 'outdir': self.outdir_method, 'full_path': self.full_path_method, 'path': self.path_method, 'found': self.found_method, 'private_dir_include': self.private_dir_include_method})

    def __repr__(self) -> str:
        r = '<{} {}: {}>'
        h = self.held_object
        assert isinstance(h, build.BuildTarget)
        return r.format(self.__class__.__name__, h.get_id(), h.filename)

    @property
    def _target_object(self) -> build.BuildTarget:
        if isinstance(self.held_object, build.BothLibraries):
            return self.held_object.get_default_object()
        assert isinstance(self.held_object, build.BuildTarget)
        return self.held_object

    def is_cross(self) -> bool:
        return not self._target_object.environment.machines.matches_build_machine(self._target_object.for_machine)

    @noPosargs
    @noKwargs
    def found_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        if not (isinstance(self.held_object, build.Executable) and self.held_object.was_returned_by_find_program):
            FeatureNew.single_use('BuildTarget.found', '0.59.0', subproject=self.held_object.subproject)
        return True

    @noPosargs
    @noKwargs
    def private_dir_include_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> build.IncludeDirs:
        return build.IncludeDirs('', [], False, [self.interpreter.backend.get_target_private_dir(self._target_object)])

    @noPosargs
    @noKwargs
    def full_path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.interpreter.backend.get_target_filename_abs(self._target_object)

    @noPosargs
    @noKwargs
    @FeatureDeprecated('BuildTarget.path', '0.55.0', 'Use BuildTarget.full_path instead')
    def path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.interpreter.backend.get_target_filename_abs(self._target_object)

    @noPosargs
    @noKwargs
    def outdir_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.interpreter.backend.get_target_dir(self._target_object)

    @noKwargs
    @typed_pos_args('extract_objects', varargs=(mesonlib.File, str, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList))
    def extract_objects_method(self, args: T.Tuple[T.List[T.Union[mesonlib.FileOrString, 'build.GeneratedTypes']]], kwargs: TYPE_nkwargs) -> build.ExtractedObjects:
        return self._target_object.extract_objects(args[0])

    @noPosargs
    @typed_kwargs('extract_all_objects', KwargInfo('recursive', bool, default=False, since='0.46.0', not_set_warning=textwrap.dedent('                extract_all_objects called without setting recursive\n                keyword argument. Meson currently defaults to\n                non-recursive to maintain backward compatibility but\n                the default will be changed in the future.\n            ')))
    def extract_all_objects_method(self, args: T.List[TYPE_nvar], kwargs: 'kwargs.BuildTargeMethodExtractAllObjects') -> build.ExtractedObjects:
        return self._target_object.extract_all_objects(kwargs['recursive'])

    @noPosargs
    @noKwargs
    @FeatureDeprecated('BuildTarget.get_id', '1.2.0', 'This was never formally documented and does not seem to have a real world use. ' + 'See https://github.com/mesonbuild/meson/pull/6061')
    def get_id_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self._target_object.get_id()

    @FeatureNew('name', '0.54.0')
    @noPosargs
    @noKwargs
    def name_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self._target_object.name