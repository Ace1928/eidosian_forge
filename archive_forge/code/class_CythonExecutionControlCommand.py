from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonExecutionControlCommand(CythonCommand, libpython.ExecutionControlCommandBase):

    @classmethod
    def register(cls):
        return cls(cls.name, cython_info)