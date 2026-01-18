from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath
class IPIDFile(Interface):
    """
    Manages a file that remembers a process ID.
    """

    def read() -> int:
        """
        Read the process ID stored in this PID file.

        @return: The contained process ID.

        @raise NoPIDFound: If this PID file does not exist.
        @raise EnvironmentError: If this PID file cannot be read.
        @raise ValueError: If this PID file's content is invalid.
        """

    def writeRunningPID() -> None:
        """
        Store the PID of the current process in this PID file.

        @raise EnvironmentError: If this PID file cannot be written.
        """

    def remove() -> None:
        """
        Remove this PID file.

        @raise EnvironmentError: If this PID file cannot be removed.
        """

    def isRunning() -> bool:
        """
        Determine whether there is a running process corresponding to the PID
        in this PID file.

        @return: True if this PID file contains a PID and a process with that
            PID is currently running; false otherwise.

        @raise EnvironmentError: If this PID file cannot be read.
        @raise InvalidPIDFileError: If this PID file's content is invalid.
        @raise StalePIDFileError: If this PID file's content refers to a PID
            for which there is no corresponding running process.
        """

    def __enter__() -> 'IPIDFile':
        """
        Enter a context using this PIDFile.

        Writes the PID file with the PID of the running process.

        @raise AlreadyRunningError: A process corresponding to the PID in this
            PID file is already running.
        """

    def __exit__(excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        """
        Exit a context using this PIDFile.

        Removes the PID file.
        """