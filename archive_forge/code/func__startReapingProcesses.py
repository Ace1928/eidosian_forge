import threading
import weakref
import warnings
from inspect import iscoroutinefunction
from functools import wraps
from queue import SimpleQueue
from twisted.python import threadable
from twisted.python.runtime import platform
from twisted.python.failure import Failure
from twisted.python.log import PythonLoggingObserver, err
from twisted.internet.defer import maybeDeferred, ensureDeferred
from twisted.internet.task import LoopingCall
import wrapt
from ._util import synchronized
from ._resultstore import ResultStore
def _startReapingProcesses(self):
    """
        Start a LoopingCall that calls reapAllProcesses.
        """
    lc = LoopingCall(self._reapAllProcesses)
    lc.clock = self._reactor
    lc.start(0.1, False)