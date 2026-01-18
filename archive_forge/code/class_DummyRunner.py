import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
@attrs(frozen=True)
class DummyRunner(Runner):
    """
    Stub for L{Runner}.

    Keep track of calls to some methods without actually doing anything.
    """
    calledMethods = attrib(type=List[str], default=Factory(list))

    def killIfRequested(self) -> None:
        self.calledMethods.append('killIfRequested')

    def startLogging(self) -> None:
        self.calledMethods.append('startLogging')

    def startReactor(self) -> None:
        self.calledMethods.append('startReactor')

    def reactorExited(self) -> None:
        self.calledMethods.append('reactorExited')