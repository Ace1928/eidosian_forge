from types import TracebackType
from typing import Callable, List, Optional, Sequence, Type, TypeVar
from unittest import TestCase as PyUnitTestCase
from attrs import Factory, define
from typing_extensions import Literal
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.protocols.amp import AMP, MAX_VALUE_LENGTH
from twisted.python.failure import Failure
from twisted.python.reflect import qual
from twisted.trial._dist import managercommands
from twisted.trial.reporter import TestResult
from ..reporter import TrialFailure
from .stream import chunk, stream
def _getFailure(self, error: TrialFailure) -> Failure:
    """
        Convert a C{sys.exc_info()}-style tuple to a L{Failure}, if necessary.
        """
    if isinstance(error, tuple):
        return Failure(error[1], error[0], error[2])
    return error