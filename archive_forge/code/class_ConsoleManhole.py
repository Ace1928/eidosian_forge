import os
import sys
import termios
import tty
from twisted.conch.insults.insults import ServerProtocol
from twisted.conch.manhole import ColoredManhole
from twisted.internet import defer, protocol, reactor, stdio
from twisted.python import failure, log, reflect
class ConsoleManhole(ColoredManhole):
    """
    A manhole protocol specifically for use with L{stdio.StandardIO}.
    """

    def connectionLost(self, reason):
        """
        When the connection is lost, there is nothing more to do.  Stop the
        reactor so that the process can exit.
        """
        reactor.stop()