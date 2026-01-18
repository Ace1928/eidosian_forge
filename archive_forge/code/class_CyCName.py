from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyCName(gdb.Function, CythonBase):
    """
    Get the C name of a Cython variable in the current context.
    Examples:

        print $cy_cname("function")
        print $cy_cname("Class.method")
        print $cy_cname("module.function")
    """

    @libpython.dont_suppress_errors
    @require_cython_frame
    @gdb_function_value_to_unicode
    def invoke(self, cyname, frame=None):
        frame = frame or gdb.selected_frame()
        cname = None
        if self.is_cython_function(frame):
            cython_function = self.get_cython_function(frame)
            if cyname in cython_function.locals:
                cname = cython_function.locals[cyname].cname
            elif cyname in cython_function.module.globals:
                cname = cython_function.module.globals[cyname].cname
            else:
                qname = '%s.%s' % (cython_function.module.name, cyname)
                if qname in cython_function.module.functions:
                    cname = cython_function.module.functions[qname].cname
        if not cname:
            cname = self.cy.functions_by_qualified_name.get(cyname)
        if not cname:
            raise gdb.GdbError('No such Cython variable: %s' % cyname)
        return cname