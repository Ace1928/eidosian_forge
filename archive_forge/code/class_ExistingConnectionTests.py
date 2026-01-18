import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
class ExistingConnectionTests(TestCase, SSHCommandClientEndpointTestsMixin):
    """
    Tests for L{SSHCommandClientEndpoint} when using the C{existingConnection}
    constructor.
    """

    def setUp(self):
        """
        Configure an SSH server with password authentication enabled for a
        well-known (to the tests) account.
        """
        SSHCommandClientEndpointTestsMixin.setUp(self)
        knownHosts = KnownHostsFile(FilePath(self.mktemp()))
        knownHosts.addHostKey(self.hostname, self.factory.publicKeys[b'ssh-rsa'])
        knownHosts.addHostKey(networkString(self.serverAddress.host), self.factory.publicKeys[b'ssh-rsa'])
        self.endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, password=self.password, knownHosts=knownHosts, ui=FixedResponseUI(False))

    def create(self):
        """
        Create and return a new L{SSHCommandClientEndpoint} using the
        C{existingConnection} constructor.
        """
        factory = Factory()
        factory.protocol = Protocol
        connected = self.endpoint.connect(factory)
        channelLookup = self.realm.channelLookup.copy()
        try:
            self.realm.channelLookup[b'session'] = WorkingExecSession
            server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        finally:
            self.realm.channelLookup.clear()
            self.realm.channelLookup.update(channelLookup)
        self._server = server
        self._client = client
        self._pump = pump
        protocol = self.successResultOf(connected)
        connection = protocol.transport.conn
        return SSHCommandClientEndpoint.existingConnection(connection, b'/bin/ls -l')

    def finishConnection(self):
        """
        Give back the connection established in L{create} over which the new
        command channel being tested will exchange data.
        """
        self._pump.pump()
        self._pump.pump()
        self._pump.pump()
        self._pump.pump()
        return (self._server, self._client, self._pump)

    def assertClientTransportState(self, client, immediateClose):
        """
        Assert that the transport for the given protocol is still connected.
        L{SSHCommandClientEndpoint.existingConnection} re-uses an SSH connected
        created by some other code, so other code is responsible for cleaning
        it up.
        """
        self.assertFalse(client.transport.disconnecting)
        self.assertFalse(client.transport.aborted)