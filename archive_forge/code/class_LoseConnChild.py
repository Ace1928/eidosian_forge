import sys
from twisted.internet import protocol, stdio
from twisted.internet.error import ConnectionDone
from twisted.python import log, reflect
class LoseConnChild(protocol.Protocol):
    exitCode = 0

    def connectionMade(self):
        self.transport.loseConnection()

    def connectionLost(self, reason):
        """
        Check that C{reason} is a L{Failure} wrapping a L{ConnectionDone}
        instance and stop the reactor.  If C{reason} is wrong for some reason,
        log something about that in C{self.errorLogFile} and make sure the
        process exits with a non-zero status.
        """
        try:
            try:
                reason.trap(ConnectionDone)
            except BaseException:
                log.err(None, 'Problem with reason passed to connectionLost')
                self.exitCode = 1
        finally:
            reactor.stop()