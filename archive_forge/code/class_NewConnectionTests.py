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
class NewConnectionTests(TestCase, SSHCommandClientEndpointTestsMixin):
    """
    Tests for L{SSHCommandClientEndpoint} when using the C{newConnection}
    constructor.
    """

    def setUp(self):
        """
        Configure an SSH server with password authentication enabled for a
        well-known (to the tests) account.
        """
        SSHCommandClientEndpointTestsMixin.setUp(self)
        self.hostKeyPath = FilePath(self.mktemp())
        self.knownHosts = KnownHostsFile(self.hostKeyPath)
        self.knownHosts.addHostKey(self.hostname, self.factory.publicKeys[b'ssh-rsa'])
        self.knownHosts.addHostKey(networkString(self.serverAddress.host), self.factory.publicKeys[b'ssh-rsa'])
        self.knownHosts.save()

    def create(self):
        """
        Create and return a new L{SSHCommandClientEndpoint} using the
        C{newConnection} constructor.
        """
        return SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, password=self.password, knownHosts=self.knownHosts, ui=FixedResponseUI(False))

    def finishConnection(self):
        """
        Establish the first attempted TCP connection using the SSH server which
        C{self.factory} can create.
        """
        return self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])

    def loseConnectionToServer(self, server, client, protocol, pump):
        """
        Lose the connection to a server and pump the L{IOPump} sufficiently for
        the client to handle the lost connection. Asserts that the client
        disconnects its transport.

        @param server: The SSH server protocol over which C{protocol} is
            running.
        @type server: L{IProtocol} provider

        @param client: The SSH client protocol over which C{protocol} is
            running.
        @type client: L{IProtocol} provider

        @param protocol: The protocol created by calling connect on the ssh
            endpoint under test.
        @type protocol: L{IProtocol} provider

        @param pump: The L{IOPump} connecting client to server.
        @type pump: L{IOPump}
        """
        closed = self.record(server, protocol, 'closed', noArgs=True)
        protocol.transport.loseConnection()
        pump.pump()
        self.assertEqual([None], closed)
        pump.pump()
        client.transport.reportDisconnect()

    def assertClientTransportState(self, client, immediateClose):
        """
        Assert that the transport for the given protocol has been disconnected.
        L{SSHCommandClientEndpoint.newConnection} creates a new dedicated SSH
        connection and cleans it up after the command exits.
        """
        if immediateClose:
            self.assertTrue(client.transport.aborted)
        else:
            self.assertTrue(client.transport.disconnecting)

    def test_interface(self):
        """
        L{SSHCommandClientEndpoint} instances provide L{IStreamClientEndpoint}.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'dummy command', b'dummy user', self.hostname, self.port)
        self.assertTrue(verifyObject(IStreamClientEndpoint, endpoint))

    def test_defaultPort(self):
        """
        L{SSHCommandClientEndpoint} uses the default port number for SSH when
        the C{port} argument is not specified.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'dummy command', b'dummy user', self.hostname)
        self.assertEqual(22, endpoint._creator.port)

    def test_specifiedPort(self):
        """
        L{SSHCommandClientEndpoint} uses the C{port} argument if specified.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'dummy command', b'dummy user', self.hostname, port=2222)
        self.assertEqual(2222, endpoint._creator.port)

    def test_destination(self):
        """
        L{SSHCommandClientEndpoint} uses the L{IReactorTCP} passed to it to
        attempt a connection to the host/port address also passed to it.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, password=self.password, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        endpoint.connect(factory)
        host, port, factory, timeout, bindAddress = self.reactor.tcpClients[0]
        self.assertEqual(self.hostname, networkString(host))
        self.assertEqual(self.port, port)
        self.assertEqual(1, len(self.reactor.tcpClients))

    def test_connectionFailed(self):
        """
        If a connection cannot be established, the L{Deferred} returned by
        L{SSHCommandClientEndpoint.connect} fires with a L{Failure}
        representing the reason for the connection setup failure.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        d = endpoint.connect(factory)
        factory = self.reactor.tcpClients[0][2]
        factory.clientConnectionFailed(None, Failure(ConnectionRefusedError()))
        self.failureResultOf(d).trap(ConnectionRefusedError)

    def test_userRejectedHostKey(self):
        """
        If the L{KnownHostsFile} instance used to construct
        L{SSHCommandClientEndpoint} rejects the SSH public key presented by the
        server, the L{Deferred} returned by L{SSHCommandClientEndpoint.connect}
        fires with a L{Failure} wrapping L{UserRejectedKey}.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=KnownHostsFile(self.mktemp()), ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        f = self.failureResultOf(connected)
        f.trap(UserRejectedKey)

    def test_mismatchedHostKey(self):
        """
        If the SSH public key presented by the SSH server does not match the
        previously remembered key, as reported by the L{KnownHostsFile}
        instance use to construct the endpoint, for that server, the
        L{Deferred} returned by L{SSHCommandClientEndpoint.connect} fires with
        a L{Failure} wrapping L{HostKeyChanged}.
        """
        firstKey = Key.fromString(privateRSA_openssh).public()
        knownHosts = KnownHostsFile(FilePath(self.mktemp()))
        knownHosts.addHostKey(networkString(self.serverAddress.host), firstKey)
        differentKey = Key.fromString(privateRSA_openssh_encrypted_aes, passphrase=b'testxp').public()
        knownHosts.addHostKey(self.hostname, differentKey)
        ui = FixedResponseUI(True)
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, password=b'dummy password', knownHosts=knownHosts, ui=ui)
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        f = self.failureResultOf(connected)
        f.trap(HostKeyChanged)

    def test_connectionClosedBeforeSecure(self):
        """
        If the connection closes at any point before the SSH transport layer
        has finished key exchange (ie, gotten to the point where we may attempt
        to authenticate), the L{Deferred} returned by
        L{SSHCommandClientEndpoint.connect} fires with a L{Failure} wrapping
        the reason for the lost connection.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        d = endpoint.connect(factory)
        transport = StringTransport()
        factory = self.reactor.tcpClients[0][2]
        client = factory.buildProtocol(None)
        client.makeConnection(transport)
        client.connectionLost(Failure(ConnectionDone()))
        self.failureResultOf(d).trap(ConnectionDone)

    def test_connectionCancelledBeforeSecure(self):
        """
        If the connection is cancelled before the SSH transport layer has
        finished key exchange (ie, gotten to the point where we may attempt to
        authenticate), the L{Deferred} returned by
        L{SSHCommandClientEndpoint.connect} fires with a L{Failure} wrapping
        L{CancelledError} and the connection is aborted.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        d = endpoint.connect(factory)
        transport = AbortableFakeTransport(None, isServer=False)
        factory = self.reactor.tcpClients[0][2]
        client = factory.buildProtocol(None)
        client.makeConnection(transport)
        d.cancel()
        self.failureResultOf(d).trap(CancelledError)
        self.assertTrue(transport.aborted)
        client.connectionLost(Failure(ConnectionDone()))

    def test_connectionCancelledBeforeConnected(self):
        """
        If the connection is cancelled before it finishes connecting, the
        connection attempt is stopped.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        d = endpoint.connect(factory)
        d.cancel()
        self.failureResultOf(d).trap(ConnectingCancelledError)
        self.assertTrue(self.reactor.connectors[0].stoppedConnecting)

    def test_passwordAuthenticationFailure(self):
        """
        If the SSH server rejects the password presented during authentication,
        the L{Deferred} returned by L{SSHCommandClientEndpoint.connect} fires
        with a L{Failure} wrapping L{AuthenticationFailed}.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, password=b'dummy password', knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        self.reactor.advance(server.service.passwordDelay)
        pump.flush()
        f = self.failureResultOf(connected)
        f.trap(AuthenticationFailed)
        self.assertClientTransportState(client, False)

    def setupKeyChecker(self, portal, users):
        """
        Create an L{ISSHPrivateKey} checker which recognizes C{users} and add it
        to C{portal}.

        @param portal: A L{Portal} to which to add the checker.
        @type portal: L{Portal}

        @param users: The users and their keys the checker will recognize.  Keys
            are byte strings giving user names.  Values are byte strings giving
            OpenSSH-formatted private keys.
        @type users: L{dict}
        """
        mapping = {k: [Key.fromString(v).public()] for k, v in users.items()}
        checker = SSHPublicKeyChecker(InMemorySSHKeyDB(mapping))
        portal.registerChecker(checker)

    def test_publicKeyAuthenticationFailure(self):
        """
        If the SSH server rejects the key pair presented during authentication,
        the L{Deferred} returned by L{SSHCommandClientEndpoint.connect} fires
        with a L{Failure} wrapping L{AuthenticationFailed}.
        """
        badKey = Key.fromString(privateRSA_openssh)
        self.setupKeyChecker(self.portal, {self.user: privateDSA_openssh})
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, keys=[badKey], knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        f = self.failureResultOf(connected)
        f.trap(AuthenticationFailed)
        self.assertTrue(client.transport.disconnecting)

    def test_authenticationFallback(self):
        """
        If the SSH server does not accept any of the specified SSH keys, the
        specified password is tried.
        """
        badKey = Key.fromString(privateRSA_openssh)
        self.setupKeyChecker(self.portal, {self.user: privateDSA_openssh})
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, keys=[badKey], password=self.password, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        self.factory.attemptsBeforeDisconnect += 1
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        pump.pump()
        errors = self.flushLoggedErrors(ConchError)
        self.assertIn('unknown channel', (errors[0].value.data, errors[0].value.value))
        self.assertEqual(1, len(errors))
        f = self.failureResultOf(connected)
        f.trap(ConchError)
        self.assertEqual(b'unknown channel', f.value.value)
        self.assertTrue(client.transport.disconnecting)

    def test_publicKeyAuthentication(self):
        """
        If L{SSHCommandClientEndpoint} is initialized with any private keys, it
        will try to use them to authenticate with the SSH server.
        """
        key = Key.fromString(privateDSA_openssh)
        self.setupKeyChecker(self.portal, {self.user: privateDSA_openssh})
        self.realm.channelLookup[b'session'] = WorkingExecSession
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, keys=[key], knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        protocol = self.successResultOf(connected)
        self.assertIsNotNone(protocol.transport)

    def test_skipPasswordAuthentication(self):
        """
        If the password is not specified, L{SSHCommandClientEndpoint} doesn't
        try it as an authentication mechanism.
        """
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        pump.pump()
        f = self.failureResultOf(connected)
        f.trap(AuthenticationFailed)
        self.assertTrue(client.transport.disconnecting)

    def test_agentAuthentication(self):
        """
        If L{SSHCommandClientEndpoint} is initialized with an
        L{SSHAgentClient}, the agent is used to authenticate with the SSH
        server. Once the connection with the SSH server has concluded, the
        connection to the agent is disconnected.
        """
        key = Key.fromString(privateRSA_openssh)
        agentServer = SSHAgentServer()
        agentServer.factory = Factory()
        agentServer.factory.keys = {key.blob(): (key, b'')}
        self.setupKeyChecker(self.portal, {self.user: privateRSA_openssh})
        agentEndpoint = SingleUseMemoryEndpoint(agentServer)
        endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False), agentEndpoint=agentEndpoint)
        self.realm.channelLookup[b'session'] = WorkingExecSession
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
        for i in range(14):
            agentEndpoint.pump.pump()
            pump.pump()
        protocol = self.successResultOf(connected)
        self.assertIsNotNone(protocol.transport)
        self.loseConnectionToServer(server, client, protocol, pump)
        self.assertTrue(client.transport.disconnecting)
        self.assertTrue(agentEndpoint.pump.clientIO.disconnecting)

    def test_loseConnection(self):
        """
        The transport connected to the protocol has a C{loseConnection} method
        which causes the channel in which the command is running to close and
        the overall connection to be closed.
        """
        self.realm.channelLookup[b'session'] = WorkingExecSession
        endpoint = self.create()
        factory = Factory()
        factory.protocol = Protocol
        connected = endpoint.connect(factory)
        server, client, pump = self.finishConnection()
        protocol = self.successResultOf(connected)
        self.loseConnectionToServer(server, client, protocol, pump)
        self.assertTrue(client.transport.disconnecting)