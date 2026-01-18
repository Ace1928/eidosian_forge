import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class IdentServer(basic.LineOnlyReceiver):
    """
    The Identification Protocol (a.k.a., "ident", a.k.a., "the Ident
    Protocol") provides a means to determine the identity of a user of a
    particular TCP connection. Given a TCP port number pair, it returns a
    character string which identifies the owner of that connection on the
    server's system.

    Server authors should subclass this class and override the lookup method.
    The default implementation returns an UNKNOWN-ERROR response for every
    query.
    """

    def lineReceived(self, line):
        parts = line.split(',')
        if len(parts) != 2:
            self.invalidQuery()
        else:
            try:
                portOnServer, portOnClient = map(int, parts)
            except ValueError:
                self.invalidQuery()
            else:
                if _MIN_PORT <= portOnServer <= _MAX_PORT and _MIN_PORT <= portOnClient <= _MAX_PORT:
                    self.validQuery(portOnServer, portOnClient)
                else:
                    self._ebLookup(failure.Failure(InvalidPort()), portOnServer, portOnClient)

    def invalidQuery(self):
        self.transport.loseConnection()

    def validQuery(self, portOnServer, portOnClient):
        """
        Called when a valid query is received to look up and deliver the
        response.

        @param portOnServer: The server port from the query.
        @param portOnClient: The client port from the query.
        """
        serverAddr = (self.transport.getHost().host, portOnServer)
        clientAddr = (self.transport.getPeer().host, portOnClient)
        defer.maybeDeferred(self.lookup, serverAddr, clientAddr).addCallback(self._cbLookup, portOnServer, portOnClient).addErrback(self._ebLookup, portOnServer, portOnClient)

    def _cbLookup(self, result, sport, cport):
        sysName, userId = result
        self.sendLine('%d, %d : USERID : %s : %s' % (sport, cport, sysName, userId))

    def _ebLookup(self, failure, sport, cport):
        if failure.check(IdentError):
            self.sendLine('%d, %d : ERROR : %s' % (sport, cport, failure.value))
        else:
            log.err(failure)
            self.sendLine('%d, %d : ERROR : %s' % (sport, cport, IdentError(failure.value)))

    def lookup(self, serverAddress, clientAddress):
        """
        Lookup user information about the specified address pair.

        Return value should be a two-tuple of system name and username.
        Acceptable values for the system name may be found online at::

            U{http://www.iana.org/assignments/operating-system-names}

        This method may also raise any IdentError subclass (or IdentError
        itself) to indicate user information will not be provided for the
        given query.

        A Deferred may also be returned.

        @param serverAddress: A two-tuple representing the server endpoint
        of the address being queried.  The first element is a string holding
        a dotted-quad IP address.  The second element is an integer
        representing the port.

        @param clientAddress: Like I{serverAddress}, but represents the
        client endpoint of the address being queried.
        """
        raise IdentError()