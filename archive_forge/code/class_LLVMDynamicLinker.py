from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class LLVMDynamicLinker(GnuLikeDynamicLinkerMixin, PosixDynamicLinkerMixin, DynamicLinker):
    """Representation of LLVM's ld.lld linker.

    This is only the gnu-like linker, not the apple like or link.exe like
    linkers.
    """
    id = 'ld.lld'

    def __init__(self, exelist: T.List[str], for_machine: mesonlib.MachineChoice, prefix_arg: T.Union[str, T.List[str]], always_args: T.List[str], *, version: str='unknown version'):
        super().__init__(exelist, for_machine, prefix_arg, always_args, version=version)
        _, _, e = mesonlib.Popen_safe(self.exelist + always_args + self._apply_prefix('--allow-shlib-undefined'))
        self.has_allow_shlib_undefined = 'unknown argument: --allow-shlib-undefined' not in e and "unknown argument: '--allow-shlib-undefined'" not in e

    def get_allow_undefined_args(self) -> T.List[str]:
        if self.has_allow_shlib_undefined:
            return self._apply_prefix('--allow-shlib-undefined')
        return []

    def get_thinlto_cache_args(self, path: str) -> T.List[str]:
        return ['-Wl,--thinlto-cache-dir=' + path]

    def get_win_subsystem_args(self, value: str) -> T.List[str]:
        version = None
        if ',' in value:
            value, version = value.split(',', 1)
        if value in self._SUBSYSTEMS:
            if version is not None:
                value += f':{version}'
            return self._apply_prefix([f'--subsystem,{value}'])
        else:
            raise mesonlib.MesonBugException(f'win_subsystem: {value} not handled in lld linker. This should not be possible.')