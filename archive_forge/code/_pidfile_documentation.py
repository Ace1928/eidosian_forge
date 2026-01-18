from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath

        Store a PID in this PID file.

        @param pid: A PID to store.

        @raise EnvironmentError: If this PID file cannot be written.

        @note: This implementation always raises an L{EnvironmentError}.
        