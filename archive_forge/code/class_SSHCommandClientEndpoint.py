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
@implementer(IStreamClientEndpoint)
class SSHCommandClientEndpoint:
    """
    L{SSHCommandClientEndpoint} exposes the command-executing functionality of
    SSH servers.

    L{SSHCommandClientEndpoint} can set up a new SSH connection, authenticate
    it in any one of a number of different ways (keys, passwords, agents),
    launch a command over that connection and then associate its input and
    output with a protocol.

    It can also re-use an existing, already-authenticated SSH connection
    (perhaps one which already has some SSH channels being used for other
    purposes).  In this case it creates a new SSH channel to use to execute the
    command.  Notably this means it supports multiplexing several different
    command invocations over a single SSH connection.
    """

    def __init__(self, creator, command):
        """
        @param creator: An L{_ISSHConnectionCreator} provider which will be
            used to set up the SSH connection which will be used to run a
            command.
        @type creator: L{_ISSHConnectionCreator} provider

        @param command: The command line to execute on the SSH server.  This
            byte string is interpreted by a shell on the SSH server, so it may
            have a value like C{"ls /"}.  Take care when trying to run a
            command like C{"/Volumes/My Stuff/a-program"} - spaces (and other
            special bytes) may require escaping.
        @type command: L{bytes}

        """
        self._creator = creator
        self._command = command

    @classmethod
    def newConnection(cls, reactor, command, username, hostname, port=None, keys=None, password=None, agentEndpoint=None, knownHosts=None, ui=None):
        """
        Create and return a new endpoint which will try to create a new
        connection to an SSH server and run a command over it.  It will also
        close the connection if there are problems leading up to the command
        being executed, after the command finishes, or if the connection
        L{Deferred} is cancelled.

        @param reactor: The reactor to use to establish the connection.
        @type reactor: L{IReactorTCP} provider

        @param command: See L{__init__}'s C{command} argument.

        @param username: The username with which to authenticate to the SSH
            server.
        @type username: L{bytes}

        @param hostname: The hostname of the SSH server.
        @type hostname: L{bytes}

        @param port: The port number of the SSH server.  By default, the
            standard SSH port number is used.
        @type port: L{int}

        @param keys: Private keys with which to authenticate to the SSH server,
            if key authentication is to be attempted (otherwise L{None}).
        @type keys: L{list} of L{Key}

        @param password: The password with which to authenticate to the SSH
            server, if password authentication is to be attempted (otherwise
            L{None}).
        @type password: L{bytes} or L{None}

        @param agentEndpoint: An L{IStreamClientEndpoint} provider which may be
            used to connect to an SSH agent, if one is to be used to help with
            authentication.
        @type agentEndpoint: L{IStreamClientEndpoint} provider

        @param knownHosts: The currently known host keys, used to check the
            host key presented by the server we actually connect to.
        @type knownHosts: L{KnownHostsFile}

        @param ui: An object for interacting with users to make decisions about
            whether to accept the server host keys.  If L{None}, a L{ConsoleUI}
            connected to /dev/tty will be used; if /dev/tty is unavailable, an
            object which answers C{b"no"} to all prompts will be used.
        @type ui: L{None} or L{ConsoleUI}

        @return: A new instance of C{cls} (probably
            L{SSHCommandClientEndpoint}).
        """
        helper = _NewConnectionHelper(reactor, hostname, port, command, username, keys, password, agentEndpoint, knownHosts, ui)
        return cls(helper, command)

    @classmethod
    def existingConnection(cls, connection, command):
        """
        Create and return a new endpoint which will try to open a new channel
        on an existing SSH connection and run a command over it.  It will
        B{not} close the connection if there is a problem executing the command
        or after the command finishes.

        @param connection: An existing connection to an SSH server.
        @type connection: L{SSHConnection}

        @param command: See L{SSHCommandClientEndpoint.newConnection}'s
            C{command} parameter.
        @type command: L{bytes}

        @return: A new instance of C{cls} (probably
            L{SSHCommandClientEndpoint}).
        """
        helper = _ExistingConnectionHelper(connection)
        return cls(helper, command)

    def connect(self, protocolFactory):
        """
        Set up an SSH connection, use a channel from that connection to launch
        a command, and hook the stdin and stdout of that command up as a
        transport for a protocol created by the given factory.

        @param protocolFactory: A L{Factory} to use to create the protocol
            which will be connected to the stdin and stdout of the command on
            the SSH server.

        @return: A L{Deferred} which will fire with an error if the connection
            cannot be set up for any reason or with the protocol instance
            created by C{protocolFactory} once it has been connected to the
            command.
        """
        d = self._creator.secureConnection()
        d.addCallback(self._executeCommand, protocolFactory)
        return d

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