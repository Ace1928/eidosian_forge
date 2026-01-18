from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def _restore_signals(self):
    for sig, hdlr in self._saved_signals:
        signal.signal(sig, hdlr)
    self._saved_signals = []