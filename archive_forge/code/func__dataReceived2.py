import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
def _dataReceived2(self, server, user, version, code, port):
    """
        The second half of the SOCKS connection setup. For a SOCKSv4 packet this
        is after the server address has been extracted from the header. For a
        SOCKSv4a packet this is after the host name has been resolved.

        @type server: L{str}
        @param server: The IP address of the destination, represented as a
            dotted quad.

        @type user: L{str}
        @param user: The username associated with the connection.

        @type version: L{int}
        @param version: The SOCKS protocol version number.

        @type code: L{int}
        @param code: The command code. 1 means establish a TCP/IP stream
            connection, and 2 means establish a TCP/IP port binding.

        @type port: L{int}
        @param port: The port number associated with the connection.
        """
    assert version == 4, 'Bad version code: %s' % version
    if not self.authorize(code, server, port, user):
        self.makeReply(91)
        return
    if code == 1:
        d = self.connectClass(server, port, SOCKSv4Outgoing, self)
        d.addErrback(lambda result, self=self: self.makeReply(91))
    elif code == 2:
        d = self.listenClass(0, SOCKSv4IncomingFactory, self, server)
        d.addCallback(lambda x, self=self: self.makeReply(90, 0, x[1], x[0]))
    else:
        raise RuntimeError(f'Bad Connect Code: {code}')
    assert self.buf == b'', 'hmm, still stuff in buffer... %s' % repr(self.buf)