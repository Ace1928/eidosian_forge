import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
class SSHConnectForwardingChannel(channel.SSHChannel):
    """
    Channel used for handling server side forwarding request.
    It acts as a client for the remote forwarding destination.

    @ivar hostport: C{(host, port)} requested by client as forwarding
        destination.
    @type hostport: L{tuple} or a C{sequence}

    @ivar client: Protocol connected to the forwarding destination.
    @type client: L{protocol.Protocol}

    @ivar clientBuf: Data received while forwarding channel is not yet
        connected.
    @type clientBuf: L{bytes}

    @var  _reactor: Reactor used for TCP connections.
    @type _reactor: A reactor.

    @ivar _channelOpenDeferred: Deferred used in testing to check the
        result of C{channelOpen}.
    @type _channelOpenDeferred: L{twisted.internet.defer.Deferred}
    """
    _reactor = reactor

    def __init__(self, hostport, *args, **kw):
        channel.SSHChannel.__init__(self, *args, **kw)
        self.hostport = hostport
        self.client = None
        self.clientBuf = b''

    def channelOpen(self, specificData):
        """
        See: L{channel.SSHChannel}
        """
        self._log.info('connecting to {host}:{port}', host=self.hostport[0], port=self.hostport[1])
        ep = HostnameEndpoint(self._reactor, self.hostport[0], self.hostport[1])
        d = connectProtocol(ep, SSHForwardingClient(self))
        d.addCallbacks(self._setClient, self._close)
        self._channelOpenDeferred = d

    def _setClient(self, client):
        """
        Called when the connection was established to the forwarding
        destination.

        @param client: Client protocol connected to the forwarding destination.
        @type  client: L{protocol.Protocol}
        """
        self.client = client
        self._log.info('connected to {host}:{port}', host=self.hostport[0], port=self.hostport[1])
        if self.clientBuf:
            self.client.transport.write(self.clientBuf)
            self.clientBuf = None
        if self.client.buf[1:]:
            self.write(self.client.buf[1:])
        self.client.buf = b''

    def _close(self, reason):
        """
        Called when failed to connect to the forwarding destination.

        @param reason: Reason why connection failed.
        @type  reason: L{twisted.python.failure.Failure}
        """
        self._log.error('failed to connect to {host}:{port}: {reason}', host=self.hostport[0], port=self.hostport[1], reason=reason)
        self.loseConnection()

    def dataReceived(self, data):
        """
        See: L{channel.SSHChannel}
        """
        if self.client:
            self.client.transport.write(data)
        else:
            self.clientBuf += data

    def closed(self):
        """
        See: L{channel.SSHChannel}
        """
        if self.client:
            self._log.info('closed remote forwarding channel {id}', id=self.id)
            if self.client.channel:
                self.loseConnection()
            self.client.transport.loseConnection()
            del self.client