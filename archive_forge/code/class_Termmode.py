import tty
import termios
import fcntl
import os
from typing import IO, ContextManager, Type, List, Union, Optional
from types import TracebackType
class Termmode(ContextManager):

    def __init__(self, stream: IO, attrs: _Attr) -> None:
        self.stream = stream
        self.attrs = attrs

    def __enter__(self) -> None:
        self.original_stty = termios.tcgetattr(self.stream)
        termios.tcsetattr(self.stream, termios.TCSANOW, self.attrs)

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)