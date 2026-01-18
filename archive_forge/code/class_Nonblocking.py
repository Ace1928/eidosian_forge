import tty
import termios
import fcntl
import os
from typing import IO, ContextManager, Type, List, Union, Optional
from types import TracebackType
class Nonblocking(ContextManager):
    """
    A context manager for making an input stream nonblocking.
    """

    def __init__(self, stream: IO) -> None:
        self.stream = stream
        self.fd = self.stream.fileno()

    def __enter__(self) -> None:
        self.orig_fl = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.orig_fl | os.O_NONBLOCK)

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.orig_fl)