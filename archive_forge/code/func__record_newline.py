from io import BytesIO
from paramiko.common import (
from paramiko.util import ClosingContextManager, u
def _record_newline(self, newline):
    if not self._flags & self.FLAG_UNIVERSAL_NEWLINE:
        return
    if self.newlines is None:
        self.newlines = newline
    elif self.newlines != newline and isinstance(self.newlines, bytes):
        self.newlines = (self.newlines, newline)
    elif newline not in self.newlines:
        self.newlines += (newline,)