from zope.interface import implementer
from twisted.internet import error, interfaces, process
from twisted.python import failure, log
def _readConnectionLost(self, reason):
    self._reader = None
    p = interfaces.IHalfCloseableProtocol(self.protocol, None)
    if p:
        try:
            p.readConnectionLost()
        except BaseException:
            log.err()
            self.connectionLost(failure.Failure())
    else:
        self.connectionLost(reason)