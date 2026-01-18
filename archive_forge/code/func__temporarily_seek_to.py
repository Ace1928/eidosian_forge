import contextlib
import io
import random
import struct
import time
import dns.exception
import dns.tsig
@contextlib.contextmanager
def _temporarily_seek_to(self, where):
    current = self.output.tell()
    try:
        self.output.seek(where)
        yield
    finally:
        self.output.seek(current)