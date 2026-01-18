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
def _gather_typelib_includes_and_update_depends(state: 'ModuleState', deps: T.Sequence[T.Union[Dependency, build.BuildTarget, CustomTarget, CustomTargetIndex]], depends: T.Sequence[T.Union[build.BuildTarget, 'build.GeneratedTypes', 'FileOrString', build.StructuredSources]]) -> T.Tuple[T.List[str], T.List[T.Union[build.BuildTarget, 'build.GeneratedTypes', 'FileOrString', build.StructuredSources]]]:
    typelib_includes: T.List[str] = []
    new_depends = list(depends)
    for dep in deps:
        if isinstance(dep, InternalDependency):
            for source in dep.sources:
                if isinstance(source, GirTarget) and source not in depends:
                    new_depends.append(source)
                    subdir = os.path.join(state.environment.get_build_dir(), source.get_subdir())
                    if subdir not in typelib_includes:
                        typelib_includes.append(subdir)
        elif isinstance(dep, build.SharedLibrary):
            for g_source in dep.generated:
                if isinstance(g_source, GirTarget):
                    subdir = os.path.join(state.environment.get_build_dir(), g_source.get_subdir())
                    if subdir not in typelib_includes:
                        typelib_includes.append(subdir)
        if isinstance(dep, Dependency):
            girdir = dep.get_variable(pkgconfig='girdir', internal='girdir', default_value='')
            assert isinstance(girdir, str), 'for mypy'
            if girdir and girdir not in typelib_includes:
                typelib_includes.append(girdir)
    return (typelib_includes, new_depends)