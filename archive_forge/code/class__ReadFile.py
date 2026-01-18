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
class _ReadFile:
    """
    A weakly file-like object which can be used with L{KnownHostsFile} to
    respond in the negative to all prompts for decisions.
    """

    def __init__(self, contents):
        """
        @param contents: L{bytes} which will be returned from every C{readline}
            call.
        """
        self._contents = contents

    def write(self, data):
        """
        No-op.

        @param data: ignored
        """

    def readline(self, count=-1):
        """
        Always give back the byte string that this L{_ReadFile} was initialized
        with.

        @param count: ignored

        @return: A fixed byte-string.
        @rtype: L{bytes}
        """
        return self._contents

    def close(self):
        """
        No-op.
        """