from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def _cancelCommands(self, reason):
    """
        Cancel all the outstanding commands, making them fail with C{reason}.
        """
    while self._current:
        cmd = self._current.popleft()
        cmd.fail(reason)