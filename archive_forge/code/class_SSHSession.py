import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
class SSHSession(channel.SSHChannel):
    """
    A generalized implementation of an SSH session.

    See RFC 4254, section 6.

    The precise implementation of the various operations that the remote end
    can send is left up to the avatar, usually via an adapter to an
    interface such as L{ISession}.

    @ivar buf: a buffer for data received before making a connection to a
        client.
    @type buf: L{bytes}
    @ivar client: a protocol for communication with a shell, an application
        program, or a subsystem (see RFC 4254, section 6.5).
    @type client: L{SSHSessionProcessProtocol}
    @ivar session: an object providing concrete implementations of session
        operations.
    @type session: L{ISession}
    """
    name = b'session'

    def __init__(self, *args, **kw):
        channel.SSHChannel.__init__(self, *args, **kw)
        self.buf = b''
        self.client = None
        self.session = None

    def request_subsystem(self, data):
        subsystem, ignored = common.getNS(data)
        log.info('Asking for subsystem "{subsystem}"', subsystem=subsystem)
        client = self.avatar.lookupSubsystem(subsystem, data)
        if client:
            pp = SSHSessionProcessProtocol(self)
            proto = wrapProcessProtocol(pp)
            client.makeConnection(proto)
            pp.makeConnection(wrapProtocol(client))
            self.client = pp
            return 1
        else:
            log.error('Failed to get subsystem')
            return 0

    def request_shell(self, data):
        log.info('Getting shell')
        if not self.session:
            self.session = ISession(self.avatar)
        try:
            pp = SSHSessionProcessProtocol(self)
            self.session.openShell(pp)
        except Exception:
            log.failure('Error getting shell')
            return 0
        else:
            self.client = pp
            return 1

    def request_exec(self, data):
        if not self.session:
            self.session = ISession(self.avatar)
        f, data = common.getNS(data)
        log.info('Executing command "{f}"', f=f)
        try:
            pp = SSHSessionProcessProtocol(self)
            self.session.execCommand(pp, f)
        except Exception:
            log.failure('Error executing command "{f}"', f=f)
            return 0
        else:
            self.client = pp
            return 1

    def request_pty_req(self, data):
        if not self.session:
            self.session = ISession(self.avatar)
        term, windowSize, modes = parseRequest_pty_req(data)
        log.info('Handling pty request: {term!r} {windowSize!r}', term=term, windowSize=windowSize)
        try:
            self.session.getPty(term, windowSize, modes)
        except Exception:
            log.failure('Error handling pty request')
            return 0
        else:
            return 1

    def request_env(self, data):
        """
        Process a request to pass an environment variable.

        @param data: The environment variable name and value, each encoded
            as an SSH protocol string and concatenated.
        @type data: L{bytes}
        @return: A true value if the request to pass this environment
            variable was accepted, otherwise a false value.
        """
        if not self.session:
            self.session = ISession(self.avatar)
        if not ISessionSetEnv.providedBy(self.session):
            return 0
        name, value, data = common.getNS(data, 2)
        try:
            self.session.setEnv(name, value)
        except EnvironmentVariableNotPermitted:
            return 0
        except Exception:
            log.failure('Error setting environment variable {name}', name=name)
            return 0
        else:
            return 1

    def request_window_change(self, data):
        if not self.session:
            self.session = ISession(self.avatar)
        winSize = parseRequest_window_change(data)
        try:
            self.session.windowChanged(winSize)
        except Exception:
            log.failure('Error changing window size')
            return 0
        else:
            return 1

    def dataReceived(self, data):
        if not self.client:
            self.buf += data
            return
        self.client.transport.write(data)

    def extReceived(self, dataType, data):
        if dataType == connection.EXTENDED_DATA_STDERR:
            if self.client and hasattr(self.client.transport, 'writeErr'):
                self.client.transport.writeErr(data)
        else:
            log.warn('Weird extended data: {dataType}', dataType=dataType)

    def eofReceived(self):
        if self.session:
            self.session.eofReceived()
        elif self.client:
            self.conn.sendClose(self)

    def closed(self):
        if self.client and self.client.transport:
            self.client.transport.loseConnection()
        if self.session:
            self.session.closed()

    def loseConnection(self):
        if self.client:
            self.client.transport.loseConnection()
        channel.SSHChannel.loseConnection(self)