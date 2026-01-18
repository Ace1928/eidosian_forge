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
class CanonicalNameLoop(Exception):
    """
    An error indicating that when trying to look up a mail exchange host, a set
    of canonical name records was found which form a cycle and resolution was
    abandoned.
    """