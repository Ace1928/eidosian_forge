import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, TextIO, TypeVar
from unittest import TestCase
from zope.interface import implementer
from attrs import frozen
from typing_extensions import Protocol, TypedDict
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.internet.protocol import ProcessProtocol
from twisted.logger import Logger
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedObject
from twisted.trial._dist import (
from twisted.trial._dist.workerreporter import WorkerReporter
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader, TrialSuite
from twisted.trial.unittest import Todo
from .stream import StreamOpen, StreamReceiver, StreamWrite
def _buildFailure(self, error: WorkerException, errorClass: str, frames: List[str]) -> Failure:
    """
        Helper to build a C{Failure} with some traceback.

        @param error: An C{Exception} instance.

        @param errorClass: The class name of the C{error} class.

        @param frames: A flat list of strings representing the information need
            to approximatively rebuild C{Failure} frames.

        @return: A L{Failure} instance with enough information about a test
           error.
        """
    errorType = namedObject(errorClass)
    failure = Failure(error, errorType)
    for i in range(0, len(frames), 3):
        failure.frames.append((frames[i], frames[i + 1], int(frames[i + 2]), [], []))
    return failure