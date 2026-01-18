from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def _sanity_check_impl(self, work_dir: str, environment: 'Environment', sname: str, code: str) -> None:
    mlog.debug('Sanity testing ' + self.get_display_language() + ' compiler:', mesonlib.join_args(self.exelist))
    mlog.debug(f'Is cross compiler: {self.is_cross!s}.')
    source_name = os.path.join(work_dir, sname)
    binname = sname.rsplit('.', 1)[0]
    mode = CompileCheckMode.LINK
    if self.is_cross:
        binname += '_cross'
        if self.exe_wrapper is None:
            mode = CompileCheckMode.COMPILE
    cargs, largs = self._get_basic_compiler_args(environment, mode)
    extra_flags = cargs + self.linker_to_compiler_args(largs)
    binname += '.exe'
    binary_name = os.path.join(work_dir, binname)
    with open(source_name, 'w', encoding='utf-8') as ofile:
        ofile.write(code)
    cmdlist = self.exelist + [sname] + self.get_output_args(binname) + extra_flags
    pc, stdo, stde = mesonlib.Popen_safe(cmdlist, cwd=work_dir)
    mlog.debug('Sanity check compiler command line:', mesonlib.join_args(cmdlist))
    mlog.debug('Sanity check compile stdout:')
    mlog.debug(stdo)
    mlog.debug('-----\nSanity check compile stderr:')
    mlog.debug(stde)
    mlog.debug('-----')
    if pc.returncode != 0:
        raise mesonlib.EnvironmentException(f'Compiler {self.name_string()} cannot compile programs.')
    if self.is_cross:
        if self.exe_wrapper is None:
            return
        cmdlist = self.exe_wrapper.get_command() + [binary_name]
    else:
        cmdlist = [binary_name]
    mlog.debug('Running test binary command: ', mesonlib.join_args(cmdlist))
    try:
        pe = subprocess.run(cmdlist, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise mesonlib.EnvironmentException(f'Could not invoke sanity test executable: {e!s}.')
    if pe.returncode != 0:
        raise mesonlib.EnvironmentException(f'Executables created by {self.language} compiler {self.name_string()} are not runnable.')