from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
import os
import typing as T
from . import NewExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import build
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..coredata import BUILTIN_DIR_OPTIONS
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import D_MODULE_VERSIONS_KW, INSTALL_DIR_KW, VARIABLES_KW, NoneType
from ..interpreterbase import FeatureNew, FeatureDeprecated
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
def _process_libs(self, libs: T.List[ANY_DEP], public: bool) -> T.Tuple[T.List[T.Union[str, build.SharedLibrary, build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]], T.List[str], T.List[str]]:
    libs = mesonlib.listify(libs)
    processed_libs: T.List[T.Union[str, build.SharedLibrary, build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]] = []
    processed_reqs: T.List[str] = []
    processed_cflags: T.List[str] = []
    for obj in libs:
        if isinstance(obj, (build.CustomTarget, build.CustomTargetIndex, build.SharedLibrary, build.StaticLibrary)) and obj.get_id() in self.metadata:
            self._check_generated_pc_deprecation(obj)
            processed_reqs.append(self.metadata[obj.get_id()].filebase)
        elif isinstance(obj, dependencies.ExternalDependency) and obj.name == 'valgrind':
            pass
        elif isinstance(obj, PkgConfigDependency):
            if obj.found():
                processed_reqs.append(obj.name)
                self.add_version_reqs(obj.name, obj.version_reqs)
        elif isinstance(obj, dependencies.InternalDependency):
            if obj.found():
                if obj.objects:
                    raise mesonlib.MesonException('.pc file cannot refer to individual object files.')
                processed_libs += obj.get_link_args()
                processed_cflags += obj.get_compile_args()
                self._add_lib_dependencies(obj.libraries, obj.whole_libraries, obj.ext_deps, public, private_external_deps=True)
                self._add_uninstalled_incdirs(obj.get_include_dirs())
        elif isinstance(obj, dependencies.Dependency):
            if obj.found():
                processed_libs += obj.get_link_args()
                processed_cflags += obj.get_compile_args()
        elif isinstance(obj, build.SharedLibrary) and obj.shared_library_only:
            processed_libs.append(obj)
            self._add_uninstalled_incdirs(obj.get_include_dirs(), obj.get_subdir())
        elif isinstance(obj, (build.SharedLibrary, build.StaticLibrary)):
            processed_libs.append(obj)
            self._add_uninstalled_incdirs(obj.get_include_dirs(), obj.get_subdir())
            self._add_lib_dependencies(obj.link_targets, obj.link_whole_targets, obj.external_deps, isinstance(obj, build.StaticLibrary) and public)
        elif isinstance(obj, (build.CustomTarget, build.CustomTargetIndex)):
            if not obj.is_linkable_target():
                raise mesonlib.MesonException('library argument contains a not linkable custom_target.')
            FeatureNew.single_use('custom_target in pkgconfig.generate libraries', '0.58.0', self.state.subproject)
            processed_libs.append(obj)
        elif isinstance(obj, str):
            processed_libs.append(obj)
        else:
            raise mesonlib.MesonException(f'library argument of type {type(obj).__name__} not a string, library or dependency object.')
    return (processed_libs, processed_reqs, processed_cflags)