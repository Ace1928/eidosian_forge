import io
import sys
def _check_not_closed(self):
    if self.closed:
        raise ValueError('I/O operation on closed file')