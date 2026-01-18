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
class PyBacktraceFull(gdb.Command):
    """Display the current python frame and all the frames within its call stack (if any)"""

    def __init__(self):
        gdb.Command.__init__(self, 'py-bt-full', gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, args, from_tty):
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        while frame:
            if frame.is_python_frame():
                frame.print_summary()
            frame = frame.older()