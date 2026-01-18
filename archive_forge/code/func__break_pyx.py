from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def _break_pyx(self, name):
    modulename, _, lineno = name.partition(':')
    lineno = int(lineno)
    if modulename:
        cython_module = self.cy.cython_namespace[modulename]
    else:
        cython_module = self.get_cython_function().module
    if (cython_module.filename, lineno) in cython_module.lineno_cy2c:
        c_lineno = cython_module.lineno_cy2c[cython_module.filename, lineno]
        breakpoint = '%s:%s' % (cython_module.c_filename, c_lineno)
        gdb.execute('break ' + breakpoint)
    else:
        raise gdb.GdbError('Not a valid line number. Does it contain actual code?')