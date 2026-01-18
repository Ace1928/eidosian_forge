import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
def _processEnded(self, result):
    """
        Record process termination and cancel the timeout call if it is active.

        @type result: L{Failure <failure.Failure>}
        @param result: The reason the child process terminated.

        @rtype: L{None} or L{Failure <failure.Failure>}
        @return: None, if the process end is expected, or the reason the child
            process terminated, if the process end is unexpected.
        """
    self.done = True
    if self._timeoutCallID is not None:
        self._timeoutCallID.cancel()
        self._timeoutCallID = None
    else:
        return result