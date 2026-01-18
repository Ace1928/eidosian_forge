import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list
def _edit_macro(self, mname, macro):
    """open an editor with the macro data in a file"""
    filename = self.shell.mktempfile(macro.value)
    self.shell.hooks.editor(filename)
    mvalue = Path(filename).read_text(encoding='utf-8')
    self.shell.user_ns[mname] = Macro(mvalue)