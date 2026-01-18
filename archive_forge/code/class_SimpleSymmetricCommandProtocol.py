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
class SimpleSymmetricCommandProtocol(FactoryNotifier):
    maybeLater = None

    def __init__(self, onConnLost=None):
        amp.AMP.__init__(self)
        self.onConnLost = onConnLost

    def sendHello(self, text):
        return self.callRemote(Hello, hello=text)

    def sendUnicodeHello(self, text, translation):
        return self.callRemote(Hello, hello=text, Print=translation)
    greeted = False

    def cmdHello(self, hello, From, optional=None, Print=None, mixedCase=None, dash_arg=None, underscore_arg=None):
        assert From == self.transport.getPeer()
        if hello == THING_I_DONT_UNDERSTAND:
            raise ThingIDontUnderstandError()
        if hello.startswith(b'fuck'):
            raise UnfriendlyGreeting("Don't be a dick.")
        if hello == b'die':
            raise DeathThreat('aieeeeeeeee')
        result = dict(hello=hello)
        if Print is not None:
            result.update(dict(Print=Print))
        self.greeted = True
        return result
    Hello.responder(cmdHello)

    def cmdGetlist(self, length):
        return {'body': [dict(x=1)] * length}
    GetList.responder(cmdGetlist)

    def okiwont(self, magicWord, list=None):
        if list is None:
            response = 'list omitted'
        else:
            response = '%s accepted' % list[0]['name']
        return dict(response=response)
    DontRejectMe.responder(okiwont)

    def waitforit(self):
        self.waiting = defer.Deferred()
        return self.waiting
    WaitForever.responder(waitforit)

    def saybye(self):
        return dict(goodbye=b'everyone')
    Goodbye.responder(saybye)

    def switchToTestProtocol(self, fail=False):
        if fail:
            name = b'no-proto'
        else:
            name = b'test-proto'
        p = TestProto(self.onConnLost, SWITCH_CLIENT_DATA)
        return self.callRemote(TestSwitchProto, SingleUseFactory(p), name=name).addCallback(lambda ign: p)

    def switchit(self, name):
        if name == b'test-proto':
            return TestProto(self.onConnLost, SWITCH_SERVER_DATA)
        raise UnknownProtocol(name)
    TestSwitchProto.responder(switchit)

    def donothing(self):
        return None
    BrokenReturn.responder(donothing)