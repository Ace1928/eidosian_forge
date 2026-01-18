from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class WrapperFactory(ServerFactory):

    def buildProtocol(wrapperSelf, addr):
        protocol = factory.buildProtocol(addr)

        def connectionMade(orig=protocol.connectionMade):
            protocol.transport.startTLS(self.contextFactory)
            orig()
        protocol.connectionMade = connectionMade
        return protocol