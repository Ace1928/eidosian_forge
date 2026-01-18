from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyList(CythonCommand):
    """
    List Cython source code. To disable to customize colouring see the cy_*
    parameters.
    """
    name = 'cy list'
    command_class = gdb.COMMAND_FILES
    completer_class = gdb.COMPLETE_NONE

    @libpython.dont_suppress_errors
    def invoke(self, _, from_tty):
        sd, lineno = self.get_source_desc()
        source = sd.get_source(lineno - 5, lineno + 5, mark_line=lineno, lex_entire=True)
        print(source)