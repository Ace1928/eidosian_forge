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
class _ISSHConnectionCreator(Interface):
    """
    An L{_ISSHConnectionCreator} knows how to create SSH connections somehow.
    """

    def secureConnection():
        """
        Return a new, connected, secured, but not yet authenticated instance of
        L{twisted.conch.ssh.transport.SSHServerTransport} or
        L{twisted.conch.ssh.transport.SSHClientTransport}.
        """

    def cleanupConnection(connection, immediate):
        """
        Perform cleanup necessary for a connection object previously returned
        from this creator's C{secureConnection} method.

        @param connection: An L{twisted.conch.ssh.transport.SSHServerTransport}
            or L{twisted.conch.ssh.transport.SSHClientTransport} returned by a
            previous call to C{secureConnection}.  It is no longer needed by
            the caller of that method and may be closed or otherwise cleaned up
            as necessary.

        @param immediate: If C{True} don't wait for any network communication,
            just close the connection immediately and as aggressively as
            necessary.
        """