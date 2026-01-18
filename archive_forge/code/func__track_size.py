import contextlib
import io
import random
import struct
import time
import dns.exception
import dns.tsig
@contextlib.contextmanager
def _track_size(self):
    start = self.output.tell()
    yield start
    if self.output.tell() > self.max_size:
        self._rollback(start)
        raise dns.exception.TooBig