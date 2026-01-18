from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyGlobals(CyLocals):
    """
    List the globals from the current Cython module.
    """
    name = 'cy globals'
    command_class = gdb.COMMAND_STACK
    completer_class = gdb.COMPLETE_NONE

    @libpython.dont_suppress_errors
    @dispatch_on_frame(c_command='info variables', python_command='py-globals')
    def invoke(self, args, from_tty):
        global_python_dict = self.get_cython_globals_dict()
        module_globals = self.get_cython_function().module.globals
        max_globals_len = 0
        max_globals_dict_len = 0
        if module_globals:
            max_globals_len = len(max(module_globals, key=len))
        if global_python_dict:
            max_globals_dict_len = len(max(global_python_dict))
        max_name_length = max(max_globals_len, max_globals_dict_len)
        seen = set()
        print('Python globals:')
        for k, v in sorted(global_python_dict.items(), key=sortkey):
            v = v.get_truncated_repr(libpython.MAX_OUTPUT_LEN)
            seen.add(k)
            print('    %-*s = %s' % (max_name_length, k, v))
        print('C globals:')
        for name, cyvar in sorted(module_globals.items(), key=sortkey):
            if name not in seen:
                try:
                    value = gdb.parse_and_eval(cyvar.cname)
                except RuntimeError:
                    pass
                else:
                    if not value.is_optimized_out:
                        self.print_gdb_value(cyvar.name, value, max_name_length, '    ')