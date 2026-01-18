import io
import sys
def _check_can_read(self):
    if not self.readable():
        raise io.UnsupportedOperation('File not open for reading')