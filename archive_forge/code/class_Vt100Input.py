from __future__ import annotations
import sys
import contextlib
import io
import termios
import tty
from asyncio import AbstractEventLoop, get_running_loop
from typing import Callable, ContextManager, Generator, TextIO
from ..key_binding import KeyPress
from .base import Input
from .posix_utils import PosixStdinReader
from .vt100_parser import Vt100Parser
class Vt100Input(Input):
    """
    Vt100 input for Posix systems.
    (This uses a posix file descriptor that can be registered in the event loop.)
    """
    _fds_not_a_terminal: set[int] = set()

    def __init__(self, stdin: TextIO) -> None:
        try:
            stdin.fileno()
        except io.UnsupportedOperation as e:
            if 'idlelib.run' in sys.modules:
                raise io.UnsupportedOperation('Stdin is not a terminal. Running from Idle is not supported.') from e
            else:
                raise io.UnsupportedOperation('Stdin is not a terminal.') from e
        isatty = stdin.isatty()
        fd = stdin.fileno()
        if not isatty and fd not in Vt100Input._fds_not_a_terminal:
            msg = 'Warning: Input is not a terminal (fd=%r).\n'
            sys.stderr.write(msg % fd)
            sys.stderr.flush()
            Vt100Input._fds_not_a_terminal.add(fd)
        self.stdin = stdin
        self._fileno = stdin.fileno()
        self._buffer: list[KeyPress] = []
        self.stdin_reader = PosixStdinReader(self._fileno, encoding=stdin.encoding)
        self.vt100_parser = Vt100Parser(lambda key_press: self._buffer.append(key_press))

    def attach(self, input_ready_callback: Callable[[], None]) -> ContextManager[None]:
        """
        Return a context manager that makes this input active in the current
        event loop.
        """
        return _attached_input(self, input_ready_callback)

    def detach(self) -> ContextManager[None]:
        """
        Return a context manager that makes sure that this input is not active
        in the current event loop.
        """
        return _detached_input(self)

    def read_keys(self) -> list[KeyPress]:
        """Read list of KeyPress."""
        data = self.stdin_reader.read()
        self.vt100_parser.feed(data)
        result = self._buffer
        self._buffer = []
        return result

    def flush_keys(self) -> list[KeyPress]:
        """
        Flush pending keys and return them.
        (Used for flushing the 'escape' key.)
        """
        self.vt100_parser.flush()
        result = self._buffer
        self._buffer = []
        return result

    @property
    def closed(self) -> bool:
        return self.stdin_reader.closed

    def raw_mode(self) -> ContextManager[None]:
        return raw_mode(self.stdin.fileno())

    def cooked_mode(self) -> ContextManager[None]:
        return cooked_mode(self.stdin.fileno())

    def fileno(self) -> int:
        return self.stdin.fileno()

    def typeahead_hash(self) -> str:
        return f'fd-{self._fileno}'