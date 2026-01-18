from __future__ import absolute_import, print_function
import os
import re
import sys
import io
from . import Errors
from .StringEncoding import EncodedString
from .Scanning import PyrexScanner, FileSourceDescriptor
from .Errors import PyrexError, CompileError, error, warning
from .Symtab import ModuleScope
from .. import Utils
from . import Options
from .Options import CompilationOptions, default_options
from .CmdLine import parse_command_line
from .Lexicon import (unicode_start_ch_any, unicode_continuation_ch_any,
def _split_qualified_name(self, qualified_name, relative_import=False):
    qualified_name_parts = qualified_name.split('.')
    last_part = qualified_name_parts.pop()
    qualified_name_parts = [(p, True) for p in qualified_name_parts]
    if last_part != '__init__':
        is_package = False
        for suffix in ('.py', '.pyx'):
            path = self.search_include_directories(qualified_name, suffix=suffix, source_pos=None, source_file_path=None, sys_path=not relative_import)
            if path:
                is_package = self._is_init_file(path)
                break
        qualified_name_parts.append((last_part, is_package))
    return qualified_name_parts