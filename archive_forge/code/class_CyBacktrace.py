from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyBacktrace(CythonCommand):
    """Print the Cython stack"""
    name = 'cy bt'
    alias = 'cy backtrace'
    command_class = gdb.COMMAND_STACK
    completer_class = gdb.COMPLETE_NONE

    @libpython.dont_suppress_errors
    @require_running_program
    def invoke(self, args, from_tty):
        frame = gdb.selected_frame()
        while frame.older():
            frame = frame.older()
        print_all = args == '-a'
        index = 0
        while frame:
            try:
                is_relevant = self.is_relevant_function(frame)
            except CyGDBError:
                is_relevant = False
            if print_all or is_relevant:
                self.print_stackframe(frame, index)
            index += 1
            frame = frame.newer()