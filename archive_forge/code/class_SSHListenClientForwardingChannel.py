import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
class SSHListenClientForwardingChannel(SSHListenForwardingChannel):
    name = b'direct-tcpip'