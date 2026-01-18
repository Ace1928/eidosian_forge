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
class FutureHello(amp.Command):
    commandName = b'hello'
    arguments = [(b'hello', amp.String()), (b'optional', amp.Boolean(optional=True)), (b'print', amp.Unicode(optional=True)), (b'from', TransportPeer(optional=True)), (b'bonus', amp.String(optional=True))]
    response = [(b'hello', amp.String()), (b'print', amp.Unicode(optional=True))]
    errors = {UnfriendlyGreeting: b'UNFRIENDLY'}