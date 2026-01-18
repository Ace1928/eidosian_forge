import tty
import termios
import fcntl
import os
from typing import IO, ContextManager, Type, List, Union, Optional
from types import TracebackType
class Cbreak(ContextManager[Termmode]):

    def __init__(self, stream: IO) -> None:
        self.stream = stream

    def __enter__(self) -> Termmode:
        self.original_stty = termios.tcgetattr(self.stream)
        tty.setcbreak(self.stream, termios.TCSANOW)
        return Termmode(self.stream, self.original_stty)

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)