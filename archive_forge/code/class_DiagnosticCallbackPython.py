from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class DiagnosticCallbackPython(DiagnosticCallback):
    """
    Overrides Director class DiagnosticCallback's virtual
    `_print()` method in Python.
    """

    def __init__(self, description, printfn):
        super().__init__(description)
        self.printfn = printfn
        if g_mupdf_trace_director:
            log(f'DiagnosticCallbackPython[{self.m_description}].__init__() self={self!r} printfn={printfn!r}')

    def __del__(self):
        if g_mupdf_trace_director:
            log(f'DiagnosticCallbackPython[{self.m_description}].__del__() destructor called.')

    def _print(self, message):
        if g_mupdf_trace_director:
            log(f'DiagnosticCallbackPython[{self.m_description}]._print(): Calling self.printfn={self.printfn!r} with message={message!r}')
        try:
            self.printfn(message)
        except Exception as e:
            log(f'DiagnosticCallbackPython[{self.m_description}]._print(): Warning: exception from self.printfn={self.printfn!r}: e={e!r}')
            raise