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
@staticmethod
def _check_pxd_filename(pos, pxd_pathname, qualified_name):
    if not pxd_pathname:
        return
    pxd_filename = os.path.basename(pxd_pathname)
    if '.' in qualified_name and qualified_name == os.path.splitext(pxd_filename)[0]:
        warning(pos, "Dotted filenames ('%s') are deprecated. Please use the normal Python package directory layout." % pxd_filename, level=1)