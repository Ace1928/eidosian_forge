import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
class SSHAgentClient(protocol.Protocol):
    """
    The client side of the SSH agent protocol.  This is equivalent to
    ssh-add(1) and can be used with either ssh-agent(1) or the SSHAgentServer
    protocol, also in this package.
    """

    def __init__(self):
        self.buf = b''
        self.deferreds = []

    def dataReceived(self, data):
        self.buf += data
        while 1:
            if len(self.buf) <= 4:
                return
            packLen = struct.unpack('!L', self.buf[:4])[0]
            if len(self.buf) < 4 + packLen:
                return
            packet, self.buf = (self.buf[4:4 + packLen], self.buf[4 + packLen:])
            reqType = ord(packet[0:1])
            d = self.deferreds.pop(0)
            if reqType == AGENT_FAILURE:
                d.errback(ConchError('agent failure'))
            elif reqType == AGENT_SUCCESS:
                d.callback(b'')
            else:
                d.callback(packet)

    def sendRequest(self, reqType, data):
        pack = struct.pack('!LB', len(data) + 1, reqType) + data
        self.transport.write(pack)
        d = defer.Deferred()
        self.deferreds.append(d)
        return d

    def requestIdentities(self):
        """
        @return: A L{Deferred} which will fire with a list of all keys found in
            the SSH agent. The list of keys is comprised of (public key blob,
            comment) tuples.
        """
        d = self.sendRequest(AGENTC_REQUEST_IDENTITIES, b'')
        d.addCallback(self._cbRequestIdentities)
        return d

    def _cbRequestIdentities(self, data):
        """
        Unpack a collection of identities into a list of tuples comprised of
        public key blobs and comments.
        """
        if ord(data[0:1]) != AGENT_IDENTITIES_ANSWER:
            raise ConchError('unexpected response: %i' % ord(data[0:1]))
        numKeys = struct.unpack('!L', data[1:5])[0]
        result = []
        data = data[5:]
        for i in range(numKeys):
            blob, data = getNS(data)
            comment, data = getNS(data)
            result.append((blob, comment))
        return result

    def addIdentity(self, blob, comment=b''):
        """
        Add a private key blob to the agent's collection of keys.
        """
        req = blob
        req += NS(comment)
        return self.sendRequest(AGENTC_ADD_IDENTITY, req)

    def signData(self, blob, data):
        """
        Request that the agent sign the given C{data} with the private key
        which corresponds to the public key given by C{blob}.  The private
        key should have been added to the agent already.

        @type blob: L{bytes}
        @type data: L{bytes}
        @return: A L{Deferred} which fires with a signature for given data
            created with the given key.
        """
        req = NS(blob)
        req += NS(data)
        req += b'\x00\x00\x00\x00'
        return self.sendRequest(AGENTC_SIGN_REQUEST, req).addCallback(self._cbSignData)

    def _cbSignData(self, data):
        if ord(data[0:1]) != AGENT_SIGN_RESPONSE:
            raise ConchError('unexpected data: %i' % ord(data[0:1]))
        signature = getNS(data[1:])[0]
        return signature

    def removeIdentity(self, blob):
        """
        Remove the private key corresponding to the public key in blob from the
        running agent.
        """
        req = NS(blob)
        return self.sendRequest(AGENTC_REMOVE_IDENTITY, req)

    def removeAllIdentities(self):
        """
        Remove all keys from the running agent.
        """
        return self.sendRequest(AGENTC_REMOVE_ALL_IDENTITIES, b'')