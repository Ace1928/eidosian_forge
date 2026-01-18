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
class SimpleSymmetricProtocol(amp.AMP):

    def sendHello(self, text):
        return self.callRemoteString(b'hello', hello=text)

    def amp_HELLO(self, box):
        return amp.Box(hello=box[b'hello'])