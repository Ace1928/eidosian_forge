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
def _ebExchange(self, failure, factory, domain):
    """
        Prepare to resend messages later.

        This errback function runs when no mail exchange server for the domain
        can be found.

        @type failure: L{Failure}
        @param failure: The reason the mail exchange lookup failed.

        @type factory: L{SMTPManagedRelayerFactory}
        @param factory: A factory which can create a relayer for the mail
            exchange server.

        @type domain: L{bytes}
        @param domain: A domain.
        """
    log.err('Error setting up managed relay factory for ' + domain)
    log.err(failure)

    def setWaiting(queue, messages):
        map(queue.setWaiting, messages)
    from twisted.internet import reactor
    reactor.callLater(30, setWaiting, self.queue, self.managed[factory])
    del self.managed[factory]