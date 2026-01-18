from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
class StaleJunkError(Exception):
    """Raised when there's junk in the spinner from a previous run."""

    def __init__(self, junk):
        Exception.__init__(self, 'There was junk in the spinner from a previous run. Use clear_junk() to clear it out: %r' % (junk,))