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
def _get_link_args(self, state: 'ModuleState', lib: T.Union[build.SharedLibrary, build.StaticLibrary], depends: T.Sequence[T.Union[build.BuildTarget, 'build.GeneratedTypes', 'FileOrString', build.StructuredSources]], include_rpath: bool=False, use_gir_args: bool=False) -> T.Tuple[T.List[str], T.List[T.Union[build.BuildTarget, 'build.GeneratedTypes', 'FileOrString', build.StructuredSources]]]:
    link_command: T.List[str] = []
    new_depends = list(depends)
    if isinstance(lib, build.SharedLibrary):
        libdir = os.path.join(state.environment.get_build_dir(), state.backend.get_target_dir(lib))
        link_command.append('-L' + libdir)
        if include_rpath:
            link_command.append('-Wl,-rpath,' + libdir)
        new_depends.append(lib)
        for d in state.backend.determine_rpath_dirs(lib):
            d = os.path.join(state.environment.get_build_dir(), d)
            link_command.append('-L' + d)
            if include_rpath:
                link_command.append('-Wl,-rpath,' + d)
    if use_gir_args and self._gir_has_option('--extra-library'):
        link_command.append('--extra-library=' + lib.name)
    else:
        link_command.append('-l' + lib.name)
    return (link_command, new_depends)