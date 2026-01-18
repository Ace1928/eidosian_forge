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
class PyTupleObjectPtr(PyObjectPtr):
    _typename = 'PyTupleObject'

    def __getitem__(self, i):
        field_ob_item = self.field('ob_item')
        return field_ob_item[i]

    def proxyval(self, visited):
        if self.as_address() in visited:
            return ProxyAlreadyVisited('(...)')
        visited.add(self.as_address())
        result = tuple((PyObjectPtr.from_pyobject_ptr(self[i]).proxyval(visited) for i in safe_range(int_from_int(self.field('ob_size')))))
        return result

    def write_repr(self, out, visited):
        if self.as_address() in visited:
            out.write('(...)')
            return
        visited.add(self.as_address())
        out.write('(')
        for i in safe_range(int_from_int(self.field('ob_size'))):
            if i > 0:
                out.write(', ')
            element = PyObjectPtr.from_pyobject_ptr(self[i])
            element.write_repr(out, visited)
        if self.field('ob_size') == 1:
            out.write(',)')
        else:
            out.write(')')