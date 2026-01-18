import os
import sys
import termios
import tty
from twisted.conch.insults.insults import ServerProtocol
from twisted.conch.manhole import ColoredManhole
from twisted.internet import defer, protocol, reactor, stdio
from twisted.python import failure, log, reflect
class TerminalProcessProtocol(protocol.ProcessProtocol):

    def __init__(self, proto):
        self.proto = proto
        self.onConnection = defer.Deferred()

    def connectionMade(self):
        self.proto.makeConnection(self)
        self.onConnection.callback(None)
        self.onConnection = None

    def write(self, data):
        """
        Write to the terminal.

        @param data: Data to write.
        @type data: L{bytes}
        """
        self.transport.write(data)

    def outReceived(self, data):
        """
        Receive data from the terminal.

        @param data: Data received.
        @type data: L{bytes}
        """
        self.proto.dataReceived(data)

    def errReceived(self, data):
        """
        Report an error.

        @param data: Data to include in L{Failure}.
        @type data: L{bytes}
        """
        self.transport.loseConnection()
        if self.proto is not None:
            self.proto.connectionLost(failure.Failure(UnexpectedOutputError(data)))
            self.proto = None

    def childConnectionLost(self, childFD):
        if self.proto is not None:
            self.proto.childConnectionLost(childFD)

    def processEnded(self, reason):
        if self.proto is not None:
            self.proto.connectionLost(reason)
            self.proto = None