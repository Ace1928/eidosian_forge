import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def _cbLostConns(self, results):
    (sSuccess, sResult), (cSuccess, cResult) = results
    self.assertFalse(sSuccess)
    self.assertFalse(cSuccess)
    acceptableErrors = [SSL.Error]
    if platform.isWindows():
        from twisted.internet.error import ConnectionLost
        acceptableErrors.append(ConnectionLost)
    sResult.trap(*acceptableErrors)
    cResult.trap(*acceptableErrors)
    return self.serverPort.stopListening()