from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class DLinker(StaticLinker):

    def __init__(self, exelist: T.List[str], arch: str, *, rsp_syntax: RSPFileSyntax=RSPFileSyntax.GCC):
        super().__init__(exelist)
        self.id = exelist[0]
        self.arch = arch
        self.__rsp_syntax = rsp_syntax

    def get_std_link_args(self, env: 'Environment', is_thin: bool) -> T.List[str]:
        return ['-lib']

    def get_output_args(self, target: str) -> T.List[str]:
        return ['-of=' + target]

    def get_linker_always_args(self) -> T.List[str]:
        if mesonlib.is_windows():
            if self.arch == 'x86_64':
                return ['-m64']
            elif self.arch == 'x86_mscoff' and self.id == 'dmd':
                return ['-m32mscoff']
            return ['-m32']
        return []

    def rsp_file_syntax(self) -> RSPFileSyntax:
        return self.__rsp_syntax