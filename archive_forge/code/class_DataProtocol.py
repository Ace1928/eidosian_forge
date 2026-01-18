from twisted.internet.protocol import Protocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
class DataProtocol(Protocol):
    data = b''

    def dataReceived(self, data):
        self.data += data
        if self.data == b'hello!':
            reactor.stop()