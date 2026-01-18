from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
def _compile_moc_impl(self, state: ModuleState, kwargs: MocCompilerKwArgs) -> T.List[build.GeneratedList]:
    self._detect_tools(state, kwargs['method'])
    if not self.tools['moc'].found():
        err_msg = "{0} sources specified and couldn't find {1}, please check your qt{2} installation"
        raise MesonException(err_msg.format('MOC', f'uic-qt{self.qt_version}', self.qt_version))
    if not (kwargs['headers'] or kwargs['sources']):
        raise build.InvalidArguments('At least one of the "headers" or "sources" keyword arguments must be provided and not empty')
    inc = state.get_include_args(include_dirs=kwargs['include_directories'])
    compile_args: T.List[str] = []
    for dep in kwargs['dependencies']:
        compile_args.extend((a for a in dep.get_all_compile_args() if a.startswith(('-I', '-D'))))
        if isinstance(dep, InternalDependency):
            for incl in dep.include_directories:
                compile_args.extend((f'-I{i}' for i in incl.to_string_list(self.interpreter.source_root, self.interpreter.environment.build_dir)))
    output: T.List[build.GeneratedList] = []
    DEPFILE_ARGS: T.List[str] = ['--output-dep-file'] if self._moc_supports_depfiles else []
    arguments = kwargs['extra_args'] + DEPFILE_ARGS + inc + compile_args + ['@INPUT@', '-o', '@OUTPUT@']
    preserve_path_from = os.path.join(state.source_root, state.subdir) if kwargs['preserve_paths'] else None
    if kwargs['headers']:
        moc_gen = build.Generator(self.tools['moc'], arguments, ['moc_@BASENAME@.cpp'], depfile='moc_@BASENAME@.cpp.d', name=f'Qt{self.qt_version} moc header')
        output.append(moc_gen.process_files(kwargs['headers'], state, preserve_path_from))
    if kwargs['sources']:
        moc_gen = build.Generator(self.tools['moc'], arguments, ['@BASENAME@.moc'], depfile='@BASENAME@.moc.d', name=f'Qt{self.qt_version} moc source')
        output.append(moc_gen.process_files(kwargs['sources'], state, preserve_path_from))
    return output