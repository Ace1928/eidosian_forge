from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
class StaticProtocol(Protocol):
    """
    Protocol stand-in that maintains test state.
    """

    def __init__(self) -> None:
        self.source: Optional[address.IAddress] = None
        self.destination: Optional[address.IAddress] = None
        self.data = b''
        self.disconnected = False

    def dataReceived(self, data: bytes) -> None:
        assert self.transport
        self.source = self.transport.getPeer()
        self.destination = self.transport.getHost()
        self.data += data