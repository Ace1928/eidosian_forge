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
def _process_reqs(self, reqs: T.Sequence[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]) -> T.List[str]:
    """Returns string names of requirements"""
    processed_reqs: T.List[str] = []
    for obj in mesonlib.listify(reqs):
        if not isinstance(obj, str):
            FeatureNew.single_use('pkgconfig.generate requirement from non-string object', '0.46.0', self.state.subproject)
        if isinstance(obj, (build.CustomTarget, build.CustomTargetIndex, build.SharedLibrary, build.StaticLibrary)) and obj.get_id() in self.metadata:
            self._check_generated_pc_deprecation(obj)
            processed_reqs.append(self.metadata[obj.get_id()].filebase)
        elif isinstance(obj, PkgConfigDependency):
            if obj.found():
                processed_reqs.append(obj.name)
                self.add_version_reqs(obj.name, obj.version_reqs)
        elif isinstance(obj, str):
            name, version_req = self.split_version_req(obj)
            processed_reqs.append(name)
            self.add_version_reqs(name, [version_req] if version_req is not None else None)
        elif isinstance(obj, dependencies.Dependency) and (not obj.found()):
            pass
        elif isinstance(obj, dependencies.ExternalDependency) and obj.name == 'threads':
            pass
        else:
            raise mesonlib.MesonException(f'requires argument not a string, library with pkgconfig-generated file or pkgconfig-dependency object, got {obj!r}')
    return processed_reqs