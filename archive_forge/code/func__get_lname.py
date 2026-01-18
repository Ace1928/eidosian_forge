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
def _get_lname(self, l: T.Union[build.SharedLibrary, build.StaticLibrary, build.CustomTarget, build.CustomTargetIndex], msg: str, pcfile: str) -> str:
    if isinstance(l, (build.CustomTargetIndex, build.CustomTarget)):
        basename = os.path.basename(l.get_filename())
        name = os.path.splitext(basename)[0]
        if name.startswith('lib'):
            name = name[3:]
        return name
    if not l.name_prefix_set:
        return l.name
    if l.prefix == '' and l.name.startswith('lib'):
        return l.name[3:]
    if isinstance(l, build.SharedLibrary) and l.import_filename:
        return l.name
    mlog.warning(msg.format(l.name, 'name_prefix', l.name, pcfile))
    return l.name