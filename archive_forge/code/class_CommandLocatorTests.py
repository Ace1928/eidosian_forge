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
class CommandLocatorTests(TestCase):
    """
    The CommandLocator should enable users to specify responders to commands as
    functions that take structured objects, annotated with metadata.
    """

    def _checkSimpleGreeting(self, locatorClass, expected):
        """
        Check that a locator of type C{locatorClass} finds a responder
        for command named I{simple} and that the found responder answers
        with the C{expected} result to a C{SimpleGreeting<"ni hao", 5>}
        command.
        """
        locator = locatorClass()
        responderCallable = locator.locateResponder(b'simple')
        result = responderCallable(amp.Box(greeting=b'ni hao', cookie=b'5'))

        def done(values):
            self.assertEqual(values, amp.AmpBox(cookieplus=b'%d' % (expected,)))
        return result.addCallback(done)

    def test_responderDecorator(self):
        """
        A method on a L{CommandLocator} subclass decorated with a L{Command}
        subclass's L{responder} decorator should be returned from
        locateResponder, wrapped in logic to serialize and deserialize its
        arguments.
        """
        return self._checkSimpleGreeting(TestLocator, 8)

    def test_responderOverriding(self):
        """
        L{CommandLocator} subclasses can override a responder inherited from
        a base class by using the L{Command.responder} decorator to register
        a new responder method.
        """
        return self._checkSimpleGreeting(OverridingLocator, 9)

    def test_responderInheritance(self):
        """
        Responder lookup follows the same rules as normal method lookup
        rules, particularly with respect to inheritance.
        """
        return self._checkSimpleGreeting(InheritingLocator, 9)

    def test_lookupFunctionDeprecatedOverride(self):
        """
        Subclasses which override locateResponder under its old name,
        lookupFunction, should have the override invoked instead.  (This tests
        an AMP subclass, because in the version of the code that could invoke
        this deprecated code path, there was no L{CommandLocator}.)
        """
        locator = OverrideLocatorAMP()
        customResponderObject = self.assertWarns(PendingDeprecationWarning, 'Override locateResponder, not lookupFunction.', __file__, lambda: locator.locateResponder(b'custom'))
        self.assertEqual(locator.customResponder, customResponderObject)
        normalResponderObject = self.assertWarns(PendingDeprecationWarning, 'Override locateResponder, not lookupFunction.', __file__, lambda: locator.locateResponder(b'simple'))
        result = normalResponderObject(amp.Box(greeting=b'ni hao', cookie=b'5'))

        def done(values):
            self.assertEqual(values, amp.AmpBox(cookieplus=b'8'))
        return result.addCallback(done)

    def test_lookupFunctionDeprecatedInvoke(self):
        """
        Invoking locateResponder under its old name, lookupFunction, should
        emit a deprecation warning, but do the same thing.
        """
        locator = TestLocator()
        responderCallable = self.assertWarns(PendingDeprecationWarning, 'Call locateResponder, not lookupFunction.', __file__, lambda: locator.lookupFunction(b'simple'))
        result = responderCallable(amp.Box(greeting=b'ni hao', cookie=b'5'))

        def done(values):
            self.assertEqual(values, amp.AmpBox(cookieplus=b'8'))
        return result.addCallback(done)