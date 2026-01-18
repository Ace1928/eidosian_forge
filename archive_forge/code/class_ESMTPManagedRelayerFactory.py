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
class ESMTPManagedRelayerFactory(SMTPManagedRelayerFactory):
    """
    A factory to create an L{ESMTPManagedRelayer}.

    This factory creates a managed relayer which relays a set of messages over
    ESMTP and informs an attempt manager of its progress.

    @type protocol: callable which returns L{ESMTPManagedRelayer}
    @ivar protocol: A callable which returns a managed relayer for ESMTP.  See
        L{ESMTPManagedRelayer.__init__} for parameters to the callable.

    @ivar secret: See L{__init__}
    @ivar contextFactory: See L{__init__}
    """
    protocol = ESMTPManagedRelayer

    def __init__(self, messages, manager, secret, contextFactory, *args, **kw):
        """
        @type messages: L{list} of L{bytes}
        @param messages: The base filenames of messages to be relayed.

        @type manager: L{_AttemptManager}
        @param manager: An attempt manager.

        @type secret: L{bytes}
        @param secret: A string for the authentication challenge response.

        @type contextFactory: L{None} or
            L{ClientContextFactory <twisted.internet.ssl.ClientContextFactory>}
        @param contextFactory: An SSL context factory.

        @type args: 1-L{tuple} of (0) L{bytes} or 2-L{tuple} of
            (0) L{bytes}, (1), L{int}
        @param args: Positional arguments for L{SMTPClient.__init__}

        @param kw: Keyword arguments for L{SMTPClient.__init__}
        """
        self.secret = secret
        self.contextFactory = contextFactory
        SMTPManagedRelayerFactory.__init__(self, messages, manager, *args, **kw)

    def buildProtocol(self, addr):
        """
        Create an L{ESMTPManagedRelayer}.

        @type addr: L{IAddress <twisted.internet.interfaces.IAddress>} provider
        @param addr: The address of the ESMTP server.

        @rtype: L{ESMTPManagedRelayer}
        @return: A managed relayer for ESMTP.
        """
        s = self.secret and self.secret(addr)
        protocol = self.protocol(self.messages, self.manager, s, self.contextFactory, *self.pArgs, **self.pKwArgs)
        protocol.factory = self
        return protocol