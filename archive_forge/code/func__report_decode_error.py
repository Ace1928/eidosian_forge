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
def _report_decode_error(self, source_desc, exc):
    msg = exc.args[-1]
    position = exc.args[2]
    encoding = exc.args[0]
    line = 1
    column = idx = 0
    with io.open(source_desc.filename, 'r', encoding='iso8859-1', newline='') as f:
        for line, data in enumerate(f, 1):
            idx += len(data)
            if idx >= position:
                column = position - (idx - len(data)) + 1
                break
    return error((source_desc, line, column), 'Decoding error, missing or incorrect coding=<encoding-name> at top of source (cannot decode with encoding %r: %s)' % (encoding, msg))