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
class wrapperobject(PyObjectPtr):
    _typename = 'wrapperobject'

    def safe_name(self):
        try:
            name = self.field('descr')['d_base']['name'].string()
            return repr(name)
        except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
            return '<unknown name>'

    def safe_tp_name(self):
        try:
            return self.field('self')['ob_type']['tp_name'].string()
        except (NullPyObjectPtr, RuntimeError, UnicodeDecodeError):
            return '<unknown tp_name>'

    def safe_self_addresss(self):
        try:
            address = long(self.field('self'))
            return '%#x' % address
        except (NullPyObjectPtr, RuntimeError):
            return '<failed to get self address>'

    def proxyval(self, visited):
        name = self.safe_name()
        tp_name = self.safe_tp_name()
        self_address = self.safe_self_addresss()
        return '<method-wrapper %s of %s object at %s>' % (name, tp_name, self_address)

    def write_repr(self, out, visited):
        proxy = self.proxyval(visited)
        out.write(proxy)