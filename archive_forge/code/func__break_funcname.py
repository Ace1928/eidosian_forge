from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def _break_funcname(self, funcname):
    func = self.cy.functions_by_qualified_name.get(funcname)
    if func and func.is_initmodule_function:
        func = None
    break_funcs = [func]
    if not func:
        funcs = self.cy.functions_by_name.get(funcname) or []
        funcs = [f for f in funcs if not f.is_initmodule_function]
        if not funcs:
            gdb.execute('break ' + funcname)
            return
        if len(funcs) > 1:
            print('There are multiple such functions:')
            for idx, func in enumerate(funcs):
                print('%3d) %s' % (idx, func.qualified_name))
            while True:
                try:
                    result = input("Select a function, press 'a' for all functions or press 'q' or '^D' to quit: ")
                except EOFError:
                    return
                else:
                    if result.lower() == 'q':
                        return
                    elif result.lower() == 'a':
                        break_funcs = funcs
                        break
                    elif result.isdigit() and 0 <= int(result) < len(funcs):
                        break_funcs = [funcs[int(result)]]
                        break
                    else:
                        print('Not understood...')
        else:
            break_funcs = [funcs[0]]
    for func in break_funcs:
        gdb.execute('break %s' % func.cname)
        if func.pf_cname:
            gdb.execute('break %s' % func.pf_cname)