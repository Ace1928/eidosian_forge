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
class FetchAndRestoreError(PythonCodeExecutor):
    """
    Context manager that fetches the error indicator in the inferior and
    restores it on exit.
    """

    def __init__(self):
        self.sizeof_PyObjectPtr = gdb.lookup_type('PyObject').pointer().sizeof
        self.pointer = self.malloc(self.sizeof_PyObjectPtr * 3)
        type = self.pointer
        value = self.pointer + self.sizeof_PyObjectPtr
        traceback = self.pointer + self.sizeof_PyObjectPtr * 2
        self.errstate = (type, value, traceback)

    def __enter__(self):
        gdb.parse_and_eval('PyErr_Fetch(%d, %d, %d)' % self.errstate)

    def __exit__(self, *args):
        if gdb.parse_and_eval('(int) PyErr_Occurred()'):
            gdb.parse_and_eval('PyErr_Print()')
        pyerr_restore = 'PyErr_Restore((PyObject *) *%d,(PyObject *) *%d,(PyObject *) *%d)'
        try:
            gdb.parse_and_eval(pyerr_restore % self.errstate)
        finally:
            self.free(self.pointer)