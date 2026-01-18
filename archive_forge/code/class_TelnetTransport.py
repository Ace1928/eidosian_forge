import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
class TelnetTransport(Telnet, ProtocolTransportMixin):
    """
    @ivar protocol: An instance of the protocol to which this
    transport is connected, or None before the connection is
    established and after it is lost.

    @ivar protocolFactory: A callable which returns protocol instances
    which provide L{ITelnetProtocol}.  This will be invoked when a
    connection is established.  It is passed *protocolArgs and
    **protocolKwArgs.

    @ivar protocolArgs: A tuple of additional arguments to
    pass to protocolFactory.

    @ivar protocolKwArgs: A dictionary of additional arguments
    to pass to protocolFactory.
    """
    disconnecting = False
    protocolFactory = None
    protocol = None

    def __init__(self, protocolFactory=None, *a, **kw):
        Telnet.__init__(self)
        if protocolFactory is not None:
            self.protocolFactory = protocolFactory
            self.protocolArgs = a
            self.protocolKwArgs = kw

    def connectionMade(self):
        if self.protocolFactory is not None:
            self.protocol = self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs)
            assert ITelnetProtocol.providedBy(self.protocol)
            try:
                factory = self.factory
            except AttributeError:
                pass
            else:
                self.protocol.factory = factory
            self.protocol.makeConnection(self)

    def connectionLost(self, reason):
        Telnet.connectionLost(self, reason)
        if self.protocol is not None:
            try:
                self.protocol.connectionLost(reason)
            finally:
                del self.protocol

    def enableLocal(self, option):
        return self.protocol.enableLocal(option)

    def enableRemote(self, option):
        return self.protocol.enableRemote(option)

    def disableLocal(self, option):
        return self.protocol.disableLocal(option)

    def disableRemote(self, option):
        return self.protocol.disableRemote(option)

    def unhandledSubnegotiation(self, command, data):
        self.protocol.unhandledSubnegotiation(command, data)

    def unhandledCommand(self, command, argument):
        self.protocol.unhandledCommand(command, argument)

    def applicationDataReceived(self, data):
        self.protocol.dataReceived(data)

    def write(self, data):
        ProtocolTransportMixin.write(self, data.replace(b'\xff', b'\xff\xff'))