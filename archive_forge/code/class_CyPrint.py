from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyPrint(CythonCommand):
    """
    Print a Cython variable using 'cy-print x' or 'cy-print module.function.x'
    """
    name = 'cy print'
    command_class = gdb.COMMAND_DATA

    @libpython.dont_suppress_errors
    def invoke(self, name, from_tty):
        global_python_dict = self.get_cython_globals_dict()
        module_globals = self.get_cython_function().module.globals
        if name in global_python_dict:
            value = global_python_dict[name].get_truncated_repr(libpython.MAX_OUTPUT_LEN)
            print('%s = %s' % (name, value))
        elif name in module_globals:
            cname = module_globals[name].cname
            try:
                value = gdb.parse_and_eval(cname)
            except RuntimeError:
                print('unable to get value of %s' % name)
            else:
                if not value.is_optimized_out:
                    self.print_gdb_value(name, value)
                else:
                    print('%s is optimized out' % name)
        elif self.is_python_function():
            return gdb.execute('py-print ' + name)
        elif self.is_cython_function():
            value = self.cy.cy_cvalue.invoke(name.lstrip('*'))
            for c in name:
                if c == '*':
                    value = value.dereference()
                else:
                    break
            self.print_gdb_value(name, value)
        else:
            gdb.execute('print ' + name)

    def complete(self):
        if self.is_cython_function():
            f = self.get_cython_function()
            return list(itertools.chain(f.locals, f.globals))
        else:
            return []