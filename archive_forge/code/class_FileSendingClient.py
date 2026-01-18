from io import BytesIO
from twisted.internet import abstract, defer, protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class FileSendingClient(protocol.Protocol):

    def __init__(self, f: BytesIO) -> None:
        self.f = f

    def connectionMade(self) -> None:
        assert self.transport is not None
        s = basic.FileSender()
        d = s.beginFileTransfer(self.f, self.transport, lambda x: x)
        d.addCallback(lambda r: self.transport.loseConnection())