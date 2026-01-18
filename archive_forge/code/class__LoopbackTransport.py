import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
@implementer(interfaces.ITransport, interfaces.IConsumer)
class _LoopbackTransport:
    disconnecting = False
    producer = None

    def __init__(self, q):
        self.q = q

    def write(self, data):
        if not isinstance(data, bytes):
            raise TypeError('Can only write bytes to ITransport')
        self.q.put(data)

    def writeSequence(self, iovec):
        self.q.put(b''.join(iovec))

    def loseConnection(self):
        self.q.disconnect = True
        self.q.put(None)

    def abortConnection(self):
        """
        Abort the connection. Same as L{loseConnection}.
        """
        self.loseConnection()

    def getPeer(self):
        return _LoopbackAddress()

    def getHost(self):
        return _LoopbackAddress()

    def registerProducer(self, producer, streaming):
        assert self.producer is None
        self.producer = producer
        self.streamingProducer = streaming
        self._pollProducer()

    def unregisterProducer(self):
        assert self.producer is not None
        self.producer = None

    def _pollProducer(self):
        if self.producer is not None and (not self.streamingProducer):
            self.producer.resumeProducing()