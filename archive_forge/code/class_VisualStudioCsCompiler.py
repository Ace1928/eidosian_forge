from __future__ import annotations
import os.path, subprocess
import textwrap
import typing as T
from ..mesonlib import EnvironmentException
from ..linkers import RSPFileSyntax
from .compilers import Compiler
from .mixins.islinker import BasicLinkerIsCompilerMixin
class VisualStudioCsCompiler(CsCompiler):
    id = 'csc'

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        if is_debug:
            return ['-debug'] if self.info.is_windows() else ['-debug:portable']
        else:
            return []

    def rsp_file_syntax(self) -> 'RSPFileSyntax':
        return RSPFileSyntax.MSVC