import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
class SSHListenForwardingFactory(protocol.Factory):

    def __init__(self, connection, hostport, klass):
        self.conn = connection
        self.hostport = hostport
        self.klass = klass

    def buildProtocol(self, addr):
        channel = self.klass(conn=self.conn)
        client = SSHForwardingClient(channel)
        channel.client = client
        addrTuple = (addr.host, addr.port)
        channelOpenData = packOpen_direct_tcpip(self.hostport, addrTuple)
        self.conn.openChannel(channel, channelOpenData)
        return client