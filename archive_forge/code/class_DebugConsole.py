import sys
import traceback
from _pydevd_bundle.pydevconsole_code import InteractiveConsole, _EvalAwaitInNewEventLoop
from _pydev_bundle import _pydev_completer
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn
from _pydev_bundle.pydev_imports import Exec
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle import pydevd_save_locals
from _pydevd_bundle.pydevd_io import IOBuf
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle.pydevd_xml import make_valid_xml_value
import inspect
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
class DebugConsole(InteractiveConsole, BaseInterpreterInterface):
    """Wrapper around code.InteractiveConsole, in order to send
    errors and outputs to the debug console
    """

    @overrides(BaseInterpreterInterface.create_std_in)
    def create_std_in(self, *args, **kwargs):
        try:
            if not self.__buffer_output:
                return sys.stdin
        except:
            pass
        return _DebugConsoleStdIn()

    @overrides(InteractiveConsole.push)
    def push(self, line, frame, buffer_output=True):
        """Change built-in stdout and stderr methods by the
        new custom StdMessage.
        execute the InteractiveConsole.push.
        Change the stdout and stderr back be the original built-ins

        :param buffer_output: if False won't redirect the output.

        Return boolean (True if more input is required else False),
        output_messages and input_messages
        """
        self.__buffer_output = buffer_output
        more = False
        if buffer_output:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
        try:
            try:
                self.frame = frame
                if buffer_output:
                    out = sys.stdout = IOBuf()
                    err = sys.stderr = IOBuf()
                more = self.add_exec(line)
            except Exception:
                exc = get_exception_traceback_str()
                if buffer_output:
                    err.buflist.append('Internal Error: %s' % (exc,))
                else:
                    sys.stderr.write('Internal Error: %s\n' % (exc,))
        finally:
            self.frame = None
            frame = None
            if buffer_output:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        if buffer_output:
            return (more, out.buflist, err.buflist)
        else:
            return (more, [], [])

    @overrides(BaseInterpreterInterface.do_add_exec)
    def do_add_exec(self, line):
        return InteractiveConsole.push(self, line)

    @overrides(InteractiveConsole.runcode)
    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback.  All exceptions are caught except
        SystemExit, which is reraised.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        try:
            updated_globals = self.get_namespace()
            initial_globals = updated_globals.copy()
            updated_locals = None
            is_async = False
            if hasattr(inspect, 'CO_COROUTINE'):
                is_async = inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE
            if is_async:
                t = _EvalAwaitInNewEventLoop(code, updated_globals, updated_locals)
                t.start()
                t.join()
                update_globals_and_locals(updated_globals, initial_globals, self.frame)
                if t.exc:
                    raise t.exc[1].with_traceback(t.exc[2])
            else:
                try:
                    exec(code, updated_globals, updated_locals)
                finally:
                    update_globals_and_locals(updated_globals, initial_globals, self.frame)
        except SystemExit:
            raise
        except:
            sys.excepthook = sys.__excepthook__
            try:
                self.showtraceback()
            finally:
                sys.__excepthook__ = sys.excepthook

    def get_namespace(self):
        dbg_namespace = {}
        dbg_namespace.update(self.frame.f_globals)
        dbg_namespace.update(self.frame.f_locals)
        return dbg_namespace