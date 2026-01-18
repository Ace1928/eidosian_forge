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
def _executeCommand(self, connection, protocolFactory):
    """
        Given a secured SSH connection, try to execute a command in a new
        channel created on it and associate the result with a protocol from the
        given factory.

        @param connection: See L{SSHCommandClientEndpoint.existingConnection}'s
            C{connection} parameter.

        @param protocolFactory: See L{SSHCommandClientEndpoint.connect}'s
            C{protocolFactory} parameter.

        @return: See L{SSHCommandClientEndpoint.connect}'s return value.
        """
    commandConnected = Deferred()

    def disconnectOnFailure(passthrough):
        immediate = passthrough.check(CancelledError)
        self._creator.cleanupConnection(connection, immediate)
        return passthrough
    commandConnected.addErrback(disconnectOnFailure)
    channel = _CommandChannel(self._creator, self._command, protocolFactory, commandConnected)
    connection.openChannel(channel)
    return commandConnected