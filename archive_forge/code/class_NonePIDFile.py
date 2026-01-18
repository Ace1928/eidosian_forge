from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath
@implementer(IPIDFile)
class NonePIDFile:
    """
    PID file implementation that does nothing.

    This is meant to be used as a "active None" object in place of a PID file
    when no PID file is desired.
    """

    def __init__(self) -> None:
        pass

    def read(self) -> int:
        raise NoPIDFound('PID file does not exist')

    def _write(self, pid: int) -> None:
        """
        Store a PID in this PID file.

        @param pid: A PID to store.

        @raise EnvironmentError: If this PID file cannot be written.

        @note: This implementation always raises an L{EnvironmentError}.
        """
        raise OSError(errno.EPERM, 'Operation not permitted')

    def writeRunningPID(self) -> None:
        self._write(0)

    def remove(self) -> None:
        raise OSError(errno.ENOENT, 'No such file or directory')

    def isRunning(self) -> bool:
        return False

    def __enter__(self) -> 'NonePIDFile':
        return self

    def __exit__(self, excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        return None