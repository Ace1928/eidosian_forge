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
def _saveHostKeyFailure(self, reason):
    """
        When host key verification fails, record the reason for the failure in
        order to fire a L{Deferred} with it later.

        @param reason: The cause of the host key verification failure.
        @type reason: L{Failure}

        @return: C{reason}
        @rtype: L{Failure}
        """
    self._hostKeyFailure = reason
    return reason