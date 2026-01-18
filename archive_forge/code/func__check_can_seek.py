import io
import sys
def _check_can_seek(self):
    if not self.readable():
        raise io.UnsupportedOperation('Seeking is only supported on files open for reading')
    if not self.seekable():
        raise io.UnsupportedOperation('The underlying file object does not support seeking')