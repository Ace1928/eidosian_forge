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
def _getMessageFile(self, i):
    """
        Retrieve the size and contents of a message.

        @type i: L{bytes}
        @param i: A 1-based message index.

        @rtype: L{Deferred <defer.Deferred>} which successfully fires with
            2-L{tuple} of (E{1}) L{int}, (E{2}) file-like object
        @return: A deferred which successfully fires with the size of the
            message and a file containing the contents of the message.
        """
    try:
        msg = int(i) - 1
        if msg < 0:
            raise ValueError()
    except ValueError:
        self.failResponse('Bad message number argument')
        return defer.succeed(None)
    sizeDeferred = defer.maybeDeferred(self.mbox.listMessages, msg)

    def cbMessageSize(size):
        if not size:
            return defer.fail(_POP3MessageDeleted())
        fileDeferred = defer.maybeDeferred(self.mbox.getMessage, msg)
        fileDeferred.addCallback(lambda fObj: (size, fObj))
        return fileDeferred

    def ebMessageSomething(err):
        errcls = err.check(_POP3MessageDeleted, ValueError, IndexError)
        if errcls is _POP3MessageDeleted:
            self.failResponse('message deleted')
        elif errcls in (ValueError, IndexError):
            if errcls is IndexError:
                warnings.warn('twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
            self.failResponse('Bad message number argument')
        else:
            log.msg('Unexpected _getMessageFile failure:')
            log.err(err)
        return None
    sizeDeferred.addCallback(cbMessageSize)
    sizeDeferred.addErrback(ebMessageSomething)
    return sizeDeferred