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
def _ids(x: T.Union[str, build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> T.Iterable[str]:
    if isinstance(x, str):
        yield x
    else:
        if x.get_id() in self.metadata:
            yield self.metadata[x.get_id()].display_name
        yield x.get_id()