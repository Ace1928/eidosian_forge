from __future__ import annotations
import errno
import os
import sys
from contextlib import contextmanager
from typing import IO, Iterator, TextIO
@contextmanager
def _blocking_io(io: IO[str]) -> Iterator[None]:
    """
    Ensure that the FD for `io` is set to blocking in here.
    """
    if sys.platform == 'win32':
        yield
        return
    try:
        fd = io.fileno()
        blocking = os.get_blocking(fd)
    except:
        blocking = True
    try:
        if not blocking:
            os.set_blocking(fd, True)
        yield
    finally:
        if not blocking:
            os.set_blocking(fd, blocking)