import sys
from twisted.internet import protocol, stdio
from twisted.python import log, reflect
class ProducerChild(protocol.Protocol):
    _paused = False
    buf = b''

    def connectionLost(self, reason):
        log.msg('*****OVER*****')
        reactor.callLater(1, reactor.stop)

    def dataReceived(self, data):
        self.buf += data
        if self._paused:
            log.startLogging(sys.stderr)
            log.msg('dataReceived while transport paused!')
            self.transport.loseConnection()
        else:
            self.transport.write(data)
            if self.buf.endswith(b'\n0\n'):
                self.transport.loseConnection()
            else:
                self.pause()

    def pause(self):
        self._paused = True
        self.transport.pauseProducing()
        reactor.callLater(0.01, self.unpause)

    def unpause(self):
        self._paused = False
        self.transport.resumeProducing()