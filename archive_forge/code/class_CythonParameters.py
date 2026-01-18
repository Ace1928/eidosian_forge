from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonParameters(object):
    """
    Simple container class that might get more functionality in the distant
    future (mostly to remind us that we're dealing with parameters).
    """

    def __init__(self):
        self.complete_unqualified = CompleteUnqualifiedFunctionNames('cy_complete_unqualified', gdb.COMMAND_BREAKPOINTS, gdb.PARAM_BOOLEAN, True)
        self.colorize_code = ColorizeSourceCode('cy_colorize_code', gdb.COMMAND_FILES, gdb.PARAM_BOOLEAN, True)
        self.terminal_background = TerminalBackground('cy_terminal_background_color', gdb.COMMAND_FILES, gdb.PARAM_STRING, 'dark')