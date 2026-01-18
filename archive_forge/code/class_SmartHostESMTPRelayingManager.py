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
class SmartHostESMTPRelayingManager(SmartHostSMTPRelayingManager):
    """
    A smart host which uses ESMTP managed relayers to send messages from the
    relay queue.

    @type factory: callable which returns L{ESMTPManagedRelayerFactory}
    @ivar factory: A callable which creates a factory for creating a managed
        relayer. See L{ESMTPManagedRelayerFactory.__init__} for parameters to
        the callable.
    """
    factory = ESMTPManagedRelayerFactory