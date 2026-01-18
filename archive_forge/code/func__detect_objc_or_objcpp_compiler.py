from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def _detect_objc_or_objcpp_compiler(env: 'Environment', lang: str, for_machine: MachineChoice) -> 'Compiler':
    from . import objc, objcpp
    popen_exceptions: T.Dict[str, T.Union[Exception, str]] = {}
    compilers, ccache, exe_wrap = _get_compilers(env, lang, for_machine)
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    comp: T.Union[T.Type[objc.ObjCCompiler], T.Type[objcpp.ObjCPPCompiler]]
    for compiler in compilers:
        arg = ['--version']
        try:
            p, out, err = Popen_safe_logged(compiler + arg, msg='Detecting compiler via')
        except OSError as e:
            popen_exceptions[join_args(compiler + arg)] = e
            continue
        version = search_version(out)
        if 'Free Software Foundation' in out:
            defines = _get_gnu_compiler_defines(compiler)
            if not defines:
                popen_exceptions[join_args(compiler)] = 'no pre-processor defines'
                continue
            version = _get_gnu_version_from_defines(defines)
            comp = objc.GnuObjCCompiler if lang == 'objc' else objcpp.GnuObjCPPCompiler
            linker = guess_nix_linker(env, compiler, comp, version, for_machine)
            return comp(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, defines, linker=linker)
        if 'clang' in out:
            linker = None
            defines = _get_clang_compiler_defines(compiler)
            if not defines:
                popen_exceptions[join_args(compiler)] = 'no pre-processor defines'
                continue
            if 'Apple' in out:
                comp = objc.AppleClangObjCCompiler if lang == 'objc' else objcpp.AppleClangObjCPPCompiler
            else:
                comp = objc.ClangObjCCompiler if lang == 'objc' else objcpp.ClangObjCPPCompiler
            if 'windows' in out or env.machines[for_machine].is_windows():
                try:
                    linker = guess_win_linker(env, compiler, comp, version, for_machine)
                except MesonException:
                    pass
            if not linker:
                linker = guess_nix_linker(env, compiler, comp, version, for_machine)
            return comp(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker, defines=defines)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')