import signal
from os.path import expanduser
from struct import unpack
from zope.interface import Interface, implementer
from twisted.conch.client.agent import SSHAgentClient
from twisted.conch.client.default import _KNOWN_HOSTS
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.ssh.channel import SSHChannel
from twisted.conch.ssh.common import NS, getNS
from twisted.conch.ssh.connection import SSHConnection
from twisted.conch.ssh.keys import Key
from twisted.conch.ssh.transport import SSHClientTransport
from twisted.conch.ssh.userauth import SSHUserAuthClient
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.error import ConnectionDone, ProcessTerminated
from twisted.internet.interfaces import IStreamClientEndpoint
from twisted.internet.protocol import Factory
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
@implementer(_ISSHConnectionCreator)
class _NewConnectionHelper:
    """
    L{_NewConnectionHelper} implements L{_ISSHConnectionCreator} by
    establishing a brand new SSH connection, securing it, and authenticating.
    """
    _KNOWN_HOSTS = _KNOWN_HOSTS
    port = 22

    def __init__(self, reactor, hostname, port, command, username, keys, password, agentEndpoint, knownHosts, ui, tty=FilePath(b'/dev/tty')):
        """
        @param tty: The path of the tty device to use in case C{ui} is L{None}.
        @type tty: L{FilePath}

        @see: L{SSHCommandClientEndpoint.newConnection}
        """
        self.reactor = reactor
        self.hostname = hostname
        if port is not None:
            self.port = port
        self.command = command
        self.username = username
        self.keys = keys
        self.password = password
        self.agentEndpoint = agentEndpoint
        if knownHosts is None:
            knownHosts = self._knownHosts()
        self.knownHosts = knownHosts
        if ui is None:
            ui = ConsoleUI(self._opener)
        self.ui = ui
        self.tty = tty

    def _opener(self):
        """
        Open the tty if possible, otherwise give back a file-like object from
        which C{b"no"} can be read.

        For use as the opener argument to L{ConsoleUI}.
        """
        try:
            return self.tty.open('rb+')
        except BaseException:
            return _ReadFile(b'no')

    @classmethod
    def _knownHosts(cls):
        """

        @return: A L{KnownHostsFile} instance pointed at the user's personal
            I{known hosts} file.
        @rtype: L{KnownHostsFile}
        """
        return KnownHostsFile.fromPath(FilePath(expanduser(cls._KNOWN_HOSTS)))

    def secureConnection(self):
        """
        Create and return a new SSH connection which has been secured and on
        which authentication has already happened.

        @return: A L{Deferred} which fires with the ready-to-use connection or
            with a failure if something prevents the connection from being
            setup, secured, or authenticated.
        """
        protocol = _CommandTransport(self)
        ready = protocol.connectionReady
        sshClient = TCP4ClientEndpoint(self.reactor, nativeString(self.hostname), self.port)
        d = connectProtocol(sshClient, protocol)
        d.addCallback(lambda ignored: ready)
        return d

    def cleanupConnection(self, connection, immediate):
        """
        Clean up the connection by closing it.  The command running on the
        endpoint has ended so the connection is no longer needed.

        @param connection: The L{SSHConnection} to close.
        @type connection: L{SSHConnection}

        @param immediate: Whether to close connection immediately.
        @type immediate: L{bool}.
        """
        if immediate:
            connection.transport.transport.abortConnection()
        else:
            connection.transport.loseConnection()