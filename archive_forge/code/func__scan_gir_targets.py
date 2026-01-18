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
@staticmethod
def _scan_gir_targets(state: 'ModuleState', girtargets: T.Sequence[build.BuildTarget]) -> T.List[T.Union[str, Executable]]:
    ret: T.List[T.Union[str, Executable]] = []
    for girtarget in girtargets:
        if isinstance(girtarget, Executable):
            ret += ['--program', girtarget]
        else:
            libpath = os.path.join(girtarget.get_subdir(), girtarget.get_filename())
            build_root = state.environment.get_build_dir()
            if isinstance(girtarget, build.SharedLibrary):
                ret += ['-L{}/{}'.format(build_root, os.path.dirname(libpath))]
                libname = girtarget.get_basename()
            else:
                libname = os.path.join(f'{build_root}/{libpath}')
            ret += ['--library', libname]
            for d in state.backend.determine_rpath_dirs(girtarget):
                d = os.path.join(state.environment.get_build_dir(), d)
                ret.append('-L' + d)
    return ret