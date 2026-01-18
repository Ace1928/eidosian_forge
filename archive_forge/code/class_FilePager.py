from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
@implementer(interfaces.IConsumer)
class FilePager(Pager):
    """
    Reads a file in chunks and sends the chunks as they come.
    """

    def __init__(self, collector, fd, callback=None, *args, **kw):
        self.chunks = []
        Pager.__init__(self, collector, callback, *args, **kw)
        self.startProducing(fd)

    def startProducing(self, fd):
        self.deferred = basic.FileSender().beginFileTransfer(fd, self)
        self.deferred.addBoth(lambda x: self.stopPaging())

    def registerProducer(self, producer, streaming):
        self.producer = producer
        if not streaming:
            self.producer.resumeProducing()

    def unregisterProducer(self):
        self.producer = None

    def write(self, chunk):
        self.chunks.append(chunk)

    def sendNextPage(self):
        """
        Get the first chunk read and send it to collector.
        """
        if not self.chunks:
            return
        val = self.chunks.pop(0)
        self.producer.resumeProducing()
        self.collector.callRemote('gotPage', val, pbanswer=False)