from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PyPrint(gdb.Command):
    """Look up the given python variable name, and print it"""

    def __init__(self):
        gdb.Command.__init__(self, 'py-print', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        name = str(args)
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        pyop_frame = frame.get_pyop()
        if not pyop_frame:
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        pyop_var, scope = pyop_frame.get_var_by_name(name)
        if pyop_var:
            print('%s %r = %s' % (scope, name, pyop_var.get_truncated_repr(MAX_OUTPUT_LEN)))
        else:
            print('%r not found' % name)