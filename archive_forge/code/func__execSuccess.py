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
def _execSuccess(self, ignored):
    """
        When the request to execute the command in this channel succeeds, use
        C{protocolFactory} to build a protocol to handle the command's input
        and output and connect the protocol to a transport representing those
        streams.

        Also fire C{commandConnected} with the created protocol after it is
        connected to its transport.

        @param ignored: The (ignored) result of the execute request
        """
    self._protocol = self._protocolFactory.buildProtocol(SSHCommandAddress(self.conn.transport.transport.getPeer(), self.conn.transport.creator.username, self.conn.transport.creator.command))
    self._protocol.makeConnection(self)
    self._commandConnected.callback(self._protocol)