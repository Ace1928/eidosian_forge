from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
class ReentryError(Exception):
    """Raised when we try to re-enter a function that forbids it."""

    def __init__(self, function):
        Exception.__init__(self, '%r in not re-entrant but was called within a call to itself.' % (function,))