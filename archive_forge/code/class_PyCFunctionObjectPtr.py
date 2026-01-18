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
class PyCFunctionObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyCFunctionObject*
    (see Include/methodobject.h and Objects/methodobject.c)
    """
    _typename = 'PyCFunctionObject'

    def proxyval(self, visited):
        m_ml = self.field('m_ml')
        try:
            ml_name = m_ml['ml_name'].string()
        except UnicodeDecodeError:
            ml_name = '<ml_name:UnicodeDecodeError>'
        pyop_m_self = self.pyop_field('m_self')
        if pyop_m_self.is_null():
            return BuiltInFunctionProxy(ml_name)
        else:
            return BuiltInMethodProxy(ml_name, pyop_m_self)