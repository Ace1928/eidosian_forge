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
def _longOperation(self, d):
    """
        Stop timeouts and block further command processing while a long
        operation completes.

        @type d: L{Deferred <defer.Deferred>}
        @param d: A deferred which triggers at the completion of a long
            operation.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which triggers after command processing resumes and
            timeouts restart after the completion of a long operation.
        """
    timeOut = self.timeOut
    self.setTimeout(None)
    self.blocked = []
    d.addCallback(self._unblock)
    d.addCallback(lambda ign: self.setTimeout(timeOut))
    return d