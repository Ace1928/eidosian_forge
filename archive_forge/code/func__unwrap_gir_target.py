from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
def _unwrap_gir_target(self, girtarget: T.Union[Executable, build.StaticLibrary, build.SharedLibrary], state: 'ModuleState') -> T.Union[Executable, build.StaticLibrary, build.SharedLibrary]:
    if not isinstance(girtarget, (Executable, build.SharedLibrary, build.StaticLibrary)):
        raise MesonException(f'Gir target must be an executable or library but is "{girtarget}" of type {type(girtarget).__name__}')
    STATIC_BUILD_REQUIRED_VERSION = '>=1.58.1'
    if isinstance(girtarget, build.StaticLibrary) and (not mesonlib.version_compare(self._get_gir_dep(state)[0].get_version(), STATIC_BUILD_REQUIRED_VERSION)):
        raise MesonException('Static libraries can only be introspected with GObject-Introspection ' + STATIC_BUILD_REQUIRED_VERSION)
    return girtarget