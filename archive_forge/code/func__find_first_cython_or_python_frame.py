from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def _find_first_cython_or_python_frame(self):
    frame = gdb.selected_frame()
    while frame:
        if self.is_cython_function(frame) or self.is_python_function(frame):
            frame.select()
            return frame
        frame = frame.older()
    raise gdb.GdbError('There is no Cython or Python frame on the stack.')