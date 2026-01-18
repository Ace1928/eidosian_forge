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
class DontRejectMe(amp.Command):
    commandName = b'dontrejectme'
    arguments = [(b'magicWord', amp.Unicode()), (b'list', amp.AmpList([(b'name', amp.Unicode())], optional=True))]
    response = [(b'response', amp.Unicode())]