import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
class ValueErrorAsyncDeferredCommandTests(ValueErrorCommandTests):
    """
    Run all of the L{IndexErrorCommandTests} tests with an
    asynchronous-Deferred returning IMailbox implementation.
    """
    mailboxType = AsyncDeferredMailbox

    def _flush(self):
        """
        Fire whatever Deferreds we've built up in our mailbox.
        """
        while self.pop3Server.mbox.waiting:
            d, a = self.pop3Server.mbox.waiting.pop()
            d.callback(a)
        ValueErrorCommandTests._flush(self)