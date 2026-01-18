from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class EvaluateOrExecuteCodeMixin(object):
    """
    Evaluate or execute Python code in a Cython or Python frame. The 'evalcode'
    method evaluations Python code, prints a traceback if an exception went
    uncaught, and returns any return value as a gdb.Value (NULL on exception).
    """

    def _fill_locals_dict(self, executor, local_dict_pointer):
        """Fill a remotely allocated dict with values from the Cython C stack"""
        cython_func = self.get_cython_function()
        for name, cyvar in cython_func.locals.items():
            if cyvar.type == PythonObject and self.is_initialized(cython_func, name):
                try:
                    val = gdb.parse_and_eval(cyvar.cname)
                except RuntimeError:
                    continue
                else:
                    if val.is_optimized_out:
                        continue
                pystringp = executor.alloc_pystring(name)
                code = '\n                    (PyObject *) PyDict_SetItem(\n                        (PyObject *) %d,\n                        (PyObject *) %d,\n                        (PyObject *) %s)\n                ' % (local_dict_pointer, pystringp, cyvar.cname)
                try:
                    if gdb.parse_and_eval(code) < 0:
                        gdb.parse_and_eval('PyErr_Print()')
                        raise gdb.GdbError('Unable to execute Python code.')
                finally:
                    executor.xdecref(pystringp)

    def _find_first_cython_or_python_frame(self):
        frame = gdb.selected_frame()
        while frame:
            if self.is_cython_function(frame) or self.is_python_function(frame):
                frame.select()
                return frame
            frame = frame.older()
        raise gdb.GdbError('There is no Cython or Python frame on the stack.')

    def _evalcode_cython(self, executor, code, input_type):
        with libpython.FetchAndRestoreError():
            global_dict = gdb.parse_and_eval('(PyObject *) PyModule_GetDict(__pyx_m)')
            local_dict = gdb.parse_and_eval('(PyObject *) PyDict_New()')
            try:
                self._fill_locals_dict(executor, libpython.pointervalue(local_dict))
                result = executor.evalcode(code, input_type, global_dict, local_dict)
            finally:
                executor.xdecref(libpython.pointervalue(local_dict))
        return result

    def evalcode(self, code, input_type):
        """
        Evaluate `code` in a Python or Cython stack frame using the given
        `input_type`.
        """
        frame = self._find_first_cython_or_python_frame()
        executor = libpython.PythonCodeExecutor()
        if self.is_python_function(frame):
            return libpython._evalcode_python(executor, code, input_type)
        return self._evalcode_cython(executor, code, input_type)