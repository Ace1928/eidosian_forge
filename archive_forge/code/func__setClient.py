import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
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