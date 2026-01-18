from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
class SyntaxTB(ListTB):
    """Extension which holds some state: the last exception value"""

    def __init__(self, color_scheme='NoColor', parent=None, config=None):
        ListTB.__init__(self, color_scheme, parent=parent, config=config)
        self.last_syntax_error = None

    def __call__(self, etype, value, elist):
        self.last_syntax_error = value
        ListTB.__call__(self, etype, value, elist)

    def structured_traceback(self, etype, value, elist, tb_offset=None, context=5):
        if isinstance(value, SyntaxError) and isinstance(value.filename, str) and isinstance(value.lineno, int):
            linecache.checkcache(value.filename)
            newtext = linecache.getline(value.filename, value.lineno)
            if newtext:
                value.text = newtext
        self.last_syntax_error = value
        return super(SyntaxTB, self).structured_traceback(etype, value, elist, tb_offset=tb_offset, context=context)

    def clear_err_state(self):
        """Return the current error state and clear it"""
        e = self.last_syntax_error
        self.last_syntax_error = None
        return e

    def stb2text(self, stb):
        """Convert a structured traceback (a list) to a string."""
        return ''.join(stb)