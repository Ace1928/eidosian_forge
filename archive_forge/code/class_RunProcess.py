from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
class RunProcess(MesonInterpreterObject):

    def __init__(self, cmd: ExternalProgram, args: T.List[str], env: mesonlib.EnvironmentVariables, source_dir: str, build_dir: str, subdir: str, mesonintrospect: T.List[str], in_builddir: bool=False, check: bool=False, capture: bool=True) -> None:
        super().__init__()
        if not isinstance(cmd, ExternalProgram):
            raise AssertionError('BUG: RunProcess must be passed an ExternalProgram')
        self.capture = capture
        self.returncode, self.stdout, self.stderr = self.run_command(cmd, args, env, source_dir, build_dir, subdir, mesonintrospect, in_builddir, check)
        self.methods.update({'returncode': self.returncode_method, 'stdout': self.stdout_method, 'stderr': self.stderr_method})

    def run_command(self, cmd: ExternalProgram, args: T.List[str], env: mesonlib.EnvironmentVariables, source_dir: str, build_dir: str, subdir: str, mesonintrospect: T.List[str], in_builddir: bool, check: bool=False) -> T.Tuple[int, str, str]:
        command_array = cmd.get_command() + args
        menv = {'MESON_SOURCE_ROOT': source_dir, 'MESON_BUILD_ROOT': build_dir, 'MESON_SUBDIR': subdir, 'MESONINTROSPECT': ' '.join([shlex.quote(x) for x in mesonintrospect])}
        if in_builddir:
            cwd = os.path.join(build_dir, subdir)
        else:
            cwd = os.path.join(source_dir, subdir)
        child_env = os.environ.copy()
        child_env.update(menv)
        child_env = env.get_env(child_env)
        stdout = subprocess.PIPE if self.capture else subprocess.DEVNULL
        mlog.debug('Running command:', mesonlib.join_args(command_array))
        try:
            p, o, e = Popen_safe(command_array, stdout=stdout, env=child_env, cwd=cwd)
            if self.capture:
                mlog.debug('--- stdout ---')
                mlog.debug(o)
            else:
                o = ''
                mlog.debug('--- stdout disabled ---')
            mlog.debug('--- stderr ---')
            mlog.debug(e)
            mlog.debug('')
            if check and p.returncode != 0:
                raise InterpreterException('Command `{}` failed with status {}.'.format(mesonlib.join_args(command_array), p.returncode))
            return (p.returncode, o, e)
        except FileNotFoundError:
            raise InterpreterException('Could not execute command `%s`.' % mesonlib.join_args(command_array))

    @noPosargs
    @noKwargs
    def returncode_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> int:
        return self.returncode

    @noPosargs
    @noKwargs
    def stdout_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.stdout

    @noPosargs
    @noKwargs
    def stderr_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.stderr