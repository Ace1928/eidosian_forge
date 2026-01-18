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
class PyObjectPtrPrinter:
    """Prints a (PyObject*)"""

    def __init__(self, gdbval):
        self.gdbval = gdbval

    def to_string(self):
        pyop = PyObjectPtr.from_pyobject_ptr(self.gdbval)
        if True:
            return pyop.get_truncated_repr(MAX_OUTPUT_LEN)
        else:
            proxyval = pyop.proxyval(set())
            return stringify(proxyval)