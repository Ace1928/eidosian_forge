import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class AddedCommandProtocol(amp.AMP):
    """
    This is a protocol which responds to L{AddErrorsCommand}, and is used to
    test that inherited commands can add their own new types of errors, but
    still respond in the same way to their parents types of errors.
    """

    def resp(self, other):
        if other:
            raise OtherInheritedError()
        else:
            raise InheritedError()
    AddErrorsCommand.responder(resp)