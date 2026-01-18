import io
import sys
def _check_can_write(self):
    if not self.writable():
        raise io.UnsupportedOperation('File not open for writing')