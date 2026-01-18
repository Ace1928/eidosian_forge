from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonFunction(CythonVariable):

    def __init__(self, module, name, cname, pf_cname, qualified_name, lineno, type=CObject, is_initmodule_function='False'):
        super(CythonFunction, self).__init__(name, cname, qualified_name, type, lineno)
        self.module = module
        self.pf_cname = pf_cname
        self.is_initmodule_function = is_initmodule_function == 'True'
        self.locals = {}
        self.arguments = []
        self.step_into_functions = set()