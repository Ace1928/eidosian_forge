from zope.interface import implementer
from twisted.internet import error, interfaces, process
from twisted.python import failure, log
def _writeConnectionLost(self, reason):
    self._writer = None
    if self.disconnecting:
        self.connectionLost(reason)
        return
    p = interfaces.IHalfCloseableProtocol(self.protocol, None)
    if p:
        try:
            p.writeConnectionLost()
        except BaseException:
            log.err()
            self.connectionLost(failure.Failure())