from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
class RedirectToPyDBIoMessages(object):

    def __init__(self, out_ctx, wrap_stream, wrap_buffer, on_write=None):
        """
        :param out_ctx:
            1=stdout and 2=stderr

        :param wrap_stream:
            Either sys.stdout or sys.stderr.

        :param bool wrap_buffer:
            If True the buffer attribute (which wraps writing bytes) should be
            wrapped.

        :param callable(str) on_write:
            May be a custom callable to be called when to write something.
            If not passed the default implementation will create an io message
            and send it through the debugger.
        """
        encoding = getattr(wrap_stream, 'encoding', None)
        if not encoding:
            encoding = os.environ.get('PYTHONIOENCODING', 'utf-8')
        self.encoding = encoding
        self._out_ctx = out_ctx
        if wrap_buffer:
            self.buffer = RedirectToPyDBIoMessages(out_ctx, wrap_stream, wrap_buffer=False, on_write=on_write)
        self._on_write = on_write

    def get_pydb(self):
        return get_global_debugger()

    def flush(self):
        pass

    def write(self, s):
        if self._on_write is not None:
            self._on_write(s)
            return
        if s:
            if isinstance(s, bytes):
                s = s.decode(self.encoding, errors='replace')
            py_db = self.get_pydb()
            if py_db is not None:
                cmd = py_db.cmd_factory.make_io_message(s, self._out_ctx)
                if py_db.writer is not None:
                    py_db.writer.add_command(cmd)