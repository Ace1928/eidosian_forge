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
class IndexErrorCommandTests(CommandMixin, unittest.TestCase):
    """
    Run all of the command tests against a mailbox which raises IndexError
    when an out of bounds request is made.  This behavior will be deprecated
    shortly and then removed.
    """
    exceptionType = IndexError
    mailboxType = DummyMailbox

    def test_LISTWithBadArgument(self):
        """
        An attempt to get metadata about a message with a bad argument fails
        with an I{ERR} response even if the mailbox implementation raises
        L{IndexError}.
        """
        return CommandMixin.test_LISTWithBadArgument(self)
    test_LISTWithBadArgument.suppress = [_listMessageSuppression]

    def test_UIDLWithBadArgument(self):
        """
        An attempt to look up the UID of a message with a bad argument fails
        with an I{ERR} response even if the mailbox implementation raises
        L{IndexError}.
        """
        return CommandMixin.test_UIDLWithBadArgument(self)
    test_UIDLWithBadArgument.suppress = [_getUidlSuppression]

    def test_TOPWithBadArgument(self):
        """
        An attempt to download some of a message with a bad argument fails with
        an I{ERR} response even if the mailbox implementation raises
        L{IndexError}.
        """
        return CommandMixin.test_TOPWithBadArgument(self)
    test_TOPWithBadArgument.suppress = [_listMessageSuppression]

    def test_RETRWithBadArgument(self):
        """
        An attempt to download a message with a bad argument fails with an
        I{ERR} response even if the mailbox implementation raises
        L{IndexError}.
        """
        return CommandMixin.test_RETRWithBadArgument(self)
    test_RETRWithBadArgument.suppress = [_listMessageSuppression]