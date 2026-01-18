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
class _ExistingConnectionHelper:
    """
    L{_ExistingConnectionHelper} implements L{_ISSHConnectionCreator} by
    handing out an existing SSH connection which is supplied to its
    initializer.
    """

    def __init__(self, connection):
        """
        @param connection: See L{SSHCommandClientEndpoint.existingConnection}'s
            C{connection} parameter.
        """
        self.connection = connection

    def secureConnection(self):
        """

        @return: A L{Deferred} that fires synchronously with the
            already-established connection object.
        """
        return succeed(self.connection)

    def cleanupConnection(self, connection, immediate):
        """
        Do not do any cleanup on the connection.  Leave that responsibility to
        whatever code created it in the first place.

        @param connection: The L{SSHConnection} which will not be modified in
            any way.
        @type connection: L{SSHConnection}

        @param immediate: An argument which will be ignored.
        @type immediate: L{bool}.
        """