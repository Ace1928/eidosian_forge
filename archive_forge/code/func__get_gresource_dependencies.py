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
def _get_gresource_dependencies(state: 'ModuleState', input_file: str, source_dirs: T.List[str], dependencies: T.Sequence[T.Union[mesonlib.File, CustomTarget, CustomTargetIndex]]) -> T.Tuple[T.List[mesonlib.FileOrString], T.List[T.Union[CustomTarget, CustomTargetIndex]], T.List[str]]:
    cmd = ['glib-compile-resources', input_file, '--generate-dependencies']
    cmd += ['--sourcedir', state.subdir]
    for source_dir in source_dirs:
        cmd += ['--sourcedir', os.path.join(state.subdir, source_dir)]
    try:
        pc, stdout, stderr = Popen_safe(cmd, cwd=state.environment.get_source_dir())
    except (FileNotFoundError, PermissionError):
        raise MesonException('Could not execute glib-compile-resources.')
    if pc.returncode != 0:
        m = f'glib-compile-resources failed to get dependencies for {cmd[1]}:\n{stderr}'
        mlog.warning(m)
        raise subprocess.CalledProcessError(pc.returncode, cmd)
    raw_dep_files: T.List[str] = stdout.split('\n')[:-1]
    depends: T.List[T.Union[CustomTarget, CustomTargetIndex]] = []
    subdirs: T.List[str] = []
    dep_files: T.List[mesonlib.FileOrString] = []
    for resfile in raw_dep_files.copy():
        resbasename = os.path.basename(resfile)
        for dep in dependencies:
            if isinstance(dep, mesonlib.File):
                if dep.fname != resbasename:
                    continue
                raw_dep_files.remove(resfile)
                dep_files.append(dep)
                subdirs.append(dep.subdir)
                break
            elif isinstance(dep, (CustomTarget, CustomTargetIndex)):
                fname = None
                outputs = {(o, os.path.basename(o)) for o in dep.get_outputs()}
                for o, baseo in outputs:
                    if baseo == resbasename:
                        fname = o
                        break
                if fname is not None:
                    raw_dep_files.remove(resfile)
                    depends.append(dep)
                    subdirs.append(dep.get_subdir())
                    break
        else:
            try:
                f = mesonlib.File.from_source_file(state.environment.get_source_dir(), '.', resfile)
            except MesonException:
                raise MesonException(f'Resource "{resfile}" listed in "{input_file}" was not found. If this is a generated file, pass the target that generates it to gnome.compile_resources() using the "dependencies" keyword argument.')
            raw_dep_files.remove(resfile)
            dep_files.append(f)
    dep_files.extend(raw_dep_files)
    return (dep_files, depends, subdirs)