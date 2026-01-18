from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydevd_bundle.pydevconsole_code import InteractiveConsole
import os
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE
import traceback
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_save_locals
from _pydev_bundle.pydev_imports import Exec, _queue
import builtins as __builtin__
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
class ConsoleWriter(InteractiveInterpreter):
    skip = 0

    def __init__(self, locals=None):
        InteractiveInterpreter.__init__(self, locals)

    def write(self, data):
        if self.skip > 0:
            self.skip -= 1
        else:
            if data == 'Traceback (most recent call last):\n':
                self.skip = 1
            sys.stderr.write(data)

    def showsyntaxerror(self, filename=None):
        """Display the syntax error that just occurred."""
        type, value, tb = sys.exc_info()
        sys.last_type = type
        sys.last_value = value
        sys.last_traceback = tb
        if filename and type is SyntaxError:
            try:
                msg, (dummy_filename, lineno, offset, line) = value.args
            except ValueError:
                pass
            else:
                value = SyntaxError(msg, (filename, lineno, offset, line))
                sys.last_value = value
        list = traceback.format_exception_only(type, value)
        sys.stderr.write(''.join(list))

    def showtraceback(self, *args, **kwargs):
        """Display the exception that just occurred."""
        try:
            type, value, tb = sys.exc_info()
            sys.last_type = type
            sys.last_value = value
            sys.last_traceback = tb
            tblist = traceback.extract_tb(tb)
            del tblist[:1]
            lines = traceback.format_list(tblist)
            if lines:
                lines.insert(0, 'Traceback (most recent call last):\n')
            lines.extend(traceback.format_exception_only(type, value))
        finally:
            tblist = tb = None
        sys.stderr.write(''.join(lines))