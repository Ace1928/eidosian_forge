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
class _CommandChannel(SSHChannel):
    """
    A L{_CommandChannel} executes a command in a session channel and connects
    its input and output to an L{IProtocol} provider.

    @ivar _creator: See L{__init__}
    @ivar _command: See L{__init__}
    @ivar _protocolFactory:  See L{__init__}
    @ivar _commandConnected:  See L{__init__}
    @ivar _protocol: An L{IProtocol} provider created using C{_protocolFactory}
        which is hooked up to the running command's input and output streams.
    """
    name = b'session'
    _log = Logger()

    def __init__(self, creator, command, protocolFactory, commandConnected):
        """
        @param creator: The L{_ISSHConnectionCreator} provider which was used
            to get the connection which this channel exists on.
        @type creator: L{_ISSHConnectionCreator} provider

        @param command: The command to be executed.
        @type command: L{bytes}

        @param protocolFactory: A client factory to use to build a L{IProtocol}
            provider to use to associate with the running command.

        @param commandConnected: A L{Deferred} to use to signal that execution
            of the command has failed or that it has succeeded and the command
            is now running.
        @type commandConnected: L{Deferred}
        """
        SSHChannel.__init__(self)
        self._creator = creator
        self._command = command
        self._protocolFactory = protocolFactory
        self._commandConnected = commandConnected
        self._reason = None

    def openFailed(self, reason):
        """
        When the request to open a new channel to run this command in fails,
        fire the C{commandConnected} deferred with a failure indicating that.
        """
        self._commandConnected.errback(reason)

    def channelOpen(self, ignored):
        """
        When the request to open a new channel to run this command in succeeds,
        issue an C{"exec"} request to run the command.
        """
        command = self.conn.sendRequest(self, b'exec', NS(self._command), wantReply=True)
        command.addCallbacks(self._execSuccess, self._execFailure)

    def _execFailure(self, reason):
        """
        When the request to execute the command in this channel fails, fire the
        C{commandConnected} deferred with a failure indicating this.

        @param reason: The cause of the command execution failure.
        @type reason: L{Failure}
        """
        self._commandConnected.errback(reason)

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

    def dataReceived(self, data):
        """
        When the command's stdout data arrives over the channel, deliver it to
        the protocol instance.

        @param data: The bytes from the command's stdout.
        @type data: L{bytes}
        """
        self._protocol.dataReceived(data)

    def request_exit_status(self, data):
        """
        When the server sends the command's exit status, record it for later
        delivery to the protocol.

        @param data: The network-order four byte representation of the exit
            status of the command.
        @type data: L{bytes}
        """
        status, = unpack('>L', data)
        if status != 0:
            self._reason = ProcessTerminated(status, None, None)

    def request_exit_signal(self, data):
        """
        When the server sends the command's exit status, record it for later
        delivery to the protocol.

        @param data: The network-order four byte representation of the exit
            signal of the command.
        @type data: L{bytes}
        """
        shortSignalName, data = getNS(data)
        coreDumped, data = (bool(ord(data[0:1])), data[1:])
        errorMessage, data = getNS(data)
        languageTag, data = getNS(data)
        signalName = f'SIG{nativeString(shortSignalName)}'
        signalID = getattr(signal, signalName, -1)
        self._log.info('Process exited with signal {shortSignalName!r}; core dumped: {coreDumped}; error message: {errorMessage}; language: {languageTag!r}', shortSignalName=shortSignalName, coreDumped=coreDumped, errorMessage=errorMessage.decode('utf-8'), languageTag=languageTag)
        self._reason = ProcessTerminated(None, signalID, None)

    def closed(self):
        """
        When the channel closes, deliver disconnection notification to the
        protocol.
        """
        self._creator.cleanupConnection(self.conn, False)
        if self._reason is None:
            reason = ConnectionDone('ssh channel closed')
        else:
            reason = self._reason
        self._protocol.connectionLost(Failure(reason))