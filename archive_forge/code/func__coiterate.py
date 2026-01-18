import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def _coiterate(self, gen):
    """
        Direct the output of an iterator to the transport and arrange for
        iteration to take place.

        @type gen: iterable which yields L{bytes}
        @param gen: An iterator over strings.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which fires when the iterator finishes.
        """
    return self.schedule(_IteratorBuffer(self.transport.writeSequence, gen))