from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def _clientLost(self, client, reason):
    self.client = None
    self._isConnecting = 0
    self._isOnline = 0
    return reason