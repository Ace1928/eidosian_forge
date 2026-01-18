from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyLocals(CythonCommand):
    """
    List the locals from the current Cython frame.
    """
    name = 'cy locals'
    command_class = gdb.COMMAND_STACK
    completer_class = gdb.COMPLETE_NONE

    @libpython.dont_suppress_errors
    @dispatch_on_frame(c_command='info locals', python_command='py-locals')
    def invoke(self, args, from_tty):
        cython_function = self.get_cython_function()
        if cython_function.is_initmodule_function:
            self.cy.globals.invoke(args, from_tty)
            return
        local_cython_vars = cython_function.locals
        max_name_length = len(max(local_cython_vars, key=len))
        for name, cyvar in sorted(local_cython_vars.items(), key=sortkey):
            if self.is_initialized(self.get_cython_function(), cyvar.name):
                value = gdb.parse_and_eval(cyvar.cname)
                if not value.is_optimized_out:
                    self.print_gdb_value(cyvar.name, value, max_name_length, '')