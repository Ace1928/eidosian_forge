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
class SmartHostSMTPRelayingManager:
    """
    A smart host which uses SMTP managed relayers to send messages from the
    relay queue.

    L{checkState} must be called periodically at which time the state of the
    relay queue is checked and new relayers are created as needed.

    In order to relay a set of messages to a mail exchange server, a smart host
    creates an attempt manager and a managed relayer factory for that set of
    messages.  When a connection is made with the mail exchange server, the
    managed relayer factory creates a managed relayer to send the messages.
    The managed relayer reports on its progress to the attempt manager which,
    in turn, updates the smart host's relay queue and information about its
    managed relayers.

    @ivar queue: See L{__init__}.
    @ivar maxConnections: See L{__init__}.
    @ivar maxMessagesPerConnection: See L{__init__}.

    @type fArgs: 3-L{tuple} of (0) L{list} of L{bytes},
        (1) L{_AttemptManager}, (2) L{bytes} or 4-L{tuple} of (0) L{list}
        of L{bytes}, (1) L{_AttemptManager}, (2) L{bytes}, (3) L{int}
    @ivar fArgs: Positional arguments for
        L{SMTPManagedRelayerFactory.__init__}.

    @type fKwArgs: L{dict}
    @ivar fKwArgs: Keyword arguments for L{SMTPManagedRelayerFactory.__init__}.

    @type factory: callable which returns L{SMTPManagedRelayerFactory}
    @ivar factory: A callable which creates a factory for creating a managed
        relayer. See L{SMTPManagedRelayerFactory.__init__} for parameters to
        the callable.

    @type PORT: L{int}
    @ivar PORT: The port over which to connect to the SMTP server.

    @type mxcalc: L{None} or L{MXCalculator}
    @ivar mxcalc: A resource for mail exchange host lookups.

    @type managed: L{dict} mapping L{SMTPManagedRelayerFactory} to L{list} of
        L{bytes}
    @ivar managed: A mapping of factory for a managed relayer to
        filenames of messages the managed relayer is responsible for.
    """
    factory: Type[protocol.ClientFactory] = SMTPManagedRelayerFactory
    PORT = 25
    mxcalc = None

    def __init__(self, queue, maxConnections=2, maxMessagesPerConnection=10):
        """
        Initialize a smart host.

        The default values specify connection limits appropriate for a
        low-volume smart host.

        @type queue: L{Queue}
        @param queue: A relay queue.

        @type maxConnections: L{int}
        @param maxConnections: The maximum number of concurrent connections to
            SMTP servers.

        @type maxMessagesPerConnection: L{int}
        @param maxMessagesPerConnection: The maximum number of messages for
            which a relayer will be given responsibility.
        """
        self.maxConnections = maxConnections
        self.maxMessagesPerConnection = maxMessagesPerConnection
        self.managed = {}
        self.queue = queue
        self.fArgs = ()
        self.fKwArgs = {}

    def __getstate__(self):
        """
        Create a representation of the non-volatile state of this object.

        @rtype: L{dict} mapping L{bytes} to L{object}
        @return: The non-volatile state of the queue.
        """
        dct = self.__dict__.copy()
        del dct['managed']
        return dct

    def __setstate__(self, state):
        """
        Restore the non-volatile state of this object and recreate the volatile
        state.

        @type state: L{dict} mapping L{bytes} to L{object}
        @param state: The non-volatile state of the queue.
        """
        self.__dict__.update(state)
        self.managed = {}

    def checkState(self):
        """
        Check the state of the relay queue and, if possible, launch relayers to
        handle waiting messages.

        @rtype: L{None} or L{Deferred}
        @return: No return value if no further messages can be relayed or a
            deferred which fires when all of the SMTP connections initiated by
            this call have disconnected.
        """
        self.queue.readDirectory()
        if len(self.managed) >= self.maxConnections:
            return
        if not self.queue.hasWaiting():
            return
        return self._checkStateMX()

    def _checkStateMX(self):
        nextMessages = self.queue.getWaiting()
        nextMessages.reverse()
        exchanges = {}
        for msg in nextMessages:
            from_, to = self.queue.getEnvelope(msg)
            name, addr = email.utils.parseaddr(to)
            parts = addr.split('@', 1)
            if len(parts) != 2:
                log.err('Illegal message destination: ' + to)
                continue
            domain = parts[1]
            self.queue.setRelaying(msg)
            exchanges.setdefault(domain, []).append(self.queue.getPath(msg))
            if len(exchanges) >= self.maxConnections - len(self.managed):
                break
        if self.mxcalc is None:
            self.mxcalc = MXCalculator()
        relays = []
        for domain, msgs in exchanges.iteritems():
            manager = _AttemptManager(self, self.queue.noisy)
            factory = self.factory(msgs, manager, *self.fArgs, **self.fKwArgs)
            self.managed[factory] = map(os.path.basename, msgs)
            relayAttemptDeferred = manager.getCompletionDeferred()
            connectSetupDeferred = self.mxcalc.getMX(domain)
            connectSetupDeferred.addCallback(lambda mx: str(mx.name))
            connectSetupDeferred.addCallback(self._cbExchange, self.PORT, factory)
            connectSetupDeferred.addErrback(lambda err: (relayAttemptDeferred.errback(err), err)[1])
            connectSetupDeferred.addErrback(self._ebExchange, factory, domain)
            relays.append(relayAttemptDeferred)
        return DeferredList(relays)

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