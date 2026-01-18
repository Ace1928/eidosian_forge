import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class FakeOutput:

    def __init__(self, coderunner, on_write, real_fileobj):
        """Fakes sys.stdout or sys.stderr

        on_write should always take unicode

        fileno should be the fileno that on_write will
                output to (e.g. 1 for standard output).
        """
        self.coderunner = coderunner
        self.on_write = on_write
        self._real_fileobj = real_fileobj

    def write(self, s, *args, **kwargs):
        self.on_write(s, *args, **kwargs)
        return self.coderunner.request_from_main_context(force_refresh=True)

    def fileno(self):
        return self._real_fileobj.fileno()

    def writelines(self, l):
        for s in l:
            self.write(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    @property
    def encoding(self):
        return self._real_fileobj.encoding