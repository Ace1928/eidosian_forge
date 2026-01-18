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
class FactoryNotifier(amp.AMP):
    factory = None

    def connectionMade(self):
        if self.factory is not None:
            self.factory.theProto = self
            if hasattr(self.factory, 'onMade'):
                self.factory.onMade.callback(None)

    def emitpong(self):
        from twisted.internet.interfaces import ISSLTransport
        if not ISSLTransport.providedBy(self.transport):
            raise DeathThreat('only send secure pings over secure channels')
        return {'pinged': True}
    SecuredPing.responder(emitpong)