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
class ESMTPManagedRelayer(ManagedRelayerMixin, relay.ESMTPRelayer):
    """
    An ESMTP managed relayer.

    This managed relayer is an ESMTP client which is responsible for sending a
    set of messages and keeping an attempt manager informed about its progress.
    """

    def __init__(self, messages, manager, *args, **kw):
        """
        @type messages: L{list} of L{bytes}
        @param messages: The base filenames of messages to be relayed.

        @type manager: L{_AttemptManager}
        @param manager: An attempt manager.

        @type args: 3-L{tuple} of (0) L{bytes}, (1) L{None} or
            L{ClientContextFactory
            <twisted.internet.ssl.ClientContextFactory>}, (2) L{bytes} or
            4-L{tuple} of (0) L{bytes}, (1) L{None} or
            L{ClientContextFactory
            <twisted.internet.ssl.ClientContextFactory>}, (2) L{bytes},
            (3) L{int}
        @param args: Positional arguments for L{ESMTPClient.__init__}

        @type kw: L{dict}
        @param kw: Keyword arguments for L{ESMTPClient.__init__}
        """
        ManagedRelayerMixin.__init__(self, manager)
        relay.ESMTPRelayer.__init__(self, messages, *args, **kw)