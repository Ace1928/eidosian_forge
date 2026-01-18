import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
class SSHListenForwardingChannel(channel.SSHChannel):

    def channelOpen(self, specificData):
        self._log.info('opened forwarding channel {id}', id=self.id)
        if len(self.client.buf) > 1:
            b = self.client.buf[1:]
            self.write(b)
        self.client.buf = b''

    def openFailed(self, reason):
        self.closed()

    def dataReceived(self, data):
        self.client.transport.write(data)

    def eofReceived(self):
        self.client.transport.loseConnection()

    def closed(self):
        if hasattr(self, 'client'):
            self._log.info('closing local forwarding channel {id}', id=self.id)
            self.client.transport.loseConnection()
            del self.client