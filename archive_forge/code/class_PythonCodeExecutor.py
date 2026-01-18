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
class PythonCodeExecutor(object):
    Py_single_input = 256
    Py_file_input = 257
    Py_eval_input = 258

    def malloc(self, size):
        chunk = gdb.parse_and_eval('(void *) malloc((size_t) %d)' % size)
        pointer = pointervalue(chunk)
        if pointer == 0:
            raise gdb.GdbError('No memory could be allocated in the inferior.')
        return pointer

    def alloc_string(self, string):
        pointer = self.malloc(len(string))
        get_selected_inferior().write_memory(pointer, string)
        return pointer

    def alloc_pystring(self, string):
        stringp = self.alloc_string(string)
        PyString_FromStringAndSize = 'PyString_FromStringAndSize'
        try:
            gdb.parse_and_eval(PyString_FromStringAndSize)
        except RuntimeError:
            PyString_FromStringAndSize = 'PyUnicode%s_FromStringAndSize' % (get_inferior_unicode_postfix(),)
        try:
            result = gdb.parse_and_eval('(PyObject *) %s((char *) %d, (size_t) %d)' % (PyString_FromStringAndSize, stringp, len(string)))
        finally:
            self.free(stringp)
        pointer = pointervalue(result)
        if pointer == 0:
            raise gdb.GdbError('Unable to allocate Python string in the inferior.')
        return pointer

    def free(self, pointer):
        gdb.parse_and_eval('(void) free((void *) %d)' % pointer)

    def incref(self, pointer):
        """Increment the reference count of a Python object in the inferior."""
        gdb.parse_and_eval('Py_IncRef((PyObject *) %d)' % pointer)

    def xdecref(self, pointer):
        """Decrement the reference count of a Python object in the inferior."""
        gdb.parse_and_eval('Py_DecRef((PyObject *) %d)' % pointer)

    def evalcode(self, code, input_type, global_dict=None, local_dict=None):
        """
        Evaluate python code `code` given as a string in the inferior and
        return the result as a gdb.Value. Returns a new reference in the
        inferior.

        Of course, executing any code in the inferior may be dangerous and may
        leave the debuggee in an unsafe state or terminate it altogether.
        """
        if '\x00' in code:
            raise gdb.GdbError('String contains NUL byte.')
        code += '\x00'
        pointer = self.alloc_string(code)
        globalsp = pointervalue(global_dict)
        localsp = pointervalue(local_dict)
        if globalsp == 0 or localsp == 0:
            raise gdb.GdbError('Unable to obtain or create locals or globals.')
        code = '\n            PyRun_String(\n                (char *) %(code)d,\n                (int) %(start)d,\n                (PyObject *) %(globals)s,\n                (PyObject *) %(locals)d)\n        ' % dict(code=pointer, start=input_type, globals=globalsp, locals=localsp)
        with FetchAndRestoreError():
            try:
                pyobject_return_value = gdb.parse_and_eval(code)
            finally:
                self.free(pointer)
        return pyobject_return_value