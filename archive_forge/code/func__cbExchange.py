import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def _cbExchange(self, address, port, factory):
    """
        Initiate a connection with a mail exchange server.

        This callback function runs after mail exchange server for the domain
        has been looked up.

        @type address: L{bytes}
        @param address: The hostname of a mail exchange server.

        @type port: L{int}
        @param port: A port number.

        @type factory: L{SMTPManagedRelayerFactory}
        @param factory: A factory which can create a relayer for the mail
            exchange server.
        """
    from twisted.internet import reactor
    reactor.connectTCP(address, port, factory)