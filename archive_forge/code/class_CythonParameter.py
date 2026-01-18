from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonParameter(gdb.Parameter):
    """
    Base class for cython parameters
    """

    def __init__(self, name, command_class, parameter_class, default=None):
        self.show_doc = self.set_doc = self.__class__.__doc__
        super(CythonParameter, self).__init__(name, command_class, parameter_class)
        if default is not None:
            self.value = default

    def __bool__(self):
        return bool(self.value)
    __nonzero__ = __bool__