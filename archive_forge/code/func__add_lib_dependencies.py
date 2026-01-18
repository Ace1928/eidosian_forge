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
def _add_lib_dependencies(self, link_targets: T.Sequence[build.BuildTargetTypes], link_whole_targets: T.Sequence[T.Union[build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex]], external_deps: T.List[dependencies.Dependency], public: bool, private_external_deps: bool=False) -> None:
    add_libs = self.add_pub_libs if public else self.add_priv_libs
    for t in link_targets:
        if t.is_internal():
            assert isinstance(t, (build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex)), 'for mypy'
            self._add_link_whole(t, public)
        else:
            add_libs([t])
    for t in link_whole_targets:
        self._add_link_whole(t, public)
    if private_external_deps:
        self.add_priv_libs(T.cast('T.List[ANY_DEP]', external_deps))
    else:
        add_libs(T.cast('T.List[ANY_DEP]', external_deps))