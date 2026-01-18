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
class CommandDispatchTests(TestCase):
    """
    The AMP CommandDispatcher class dispatches converts AMP boxes into commands
    and responses using Command.responder decorator.

    Note: Originally, AMP's factoring was such that many tests for this
    functionality are now implemented as full round-trip tests in L{AMPTests}.
    Future tests should be written at this level instead, to ensure API
    compatibility and to provide more granular, readable units of test
    coverage.
    """

    def setUp(self):
        """
        Create a dispatcher to use.
        """
        self.locator = FakeLocator()
        self.sender = FakeSender()
        self.dispatcher = amp.BoxDispatcher(self.locator)
        self.dispatcher.startReceivingBoxes(self.sender)

    def test_receivedAsk(self):
        """
        L{CommandDispatcher.ampBoxReceived} should locate the appropriate
        command in its responder lookup, based on the '_ask' key.
        """
        received = []

        def thunk(box):
            received.append(box)
            return amp.Box({'hello': 'goodbye'})
        input = amp.Box(_command='hello', _ask='test-command-id', hello='world')
        self.locator.commands['hello'] = thunk
        self.dispatcher.ampBoxReceived(input)
        self.assertEqual(received, [input])

    def test_sendUnhandledError(self):
        """
        L{CommandDispatcher} should relay its unhandled errors in responding to
        boxes to its boxSender.
        """
        err = RuntimeError('something went wrong, oh no')
        self.sender.expectError()
        self.dispatcher.unhandledError(Failure(err))
        self.assertEqual(len(self.sender.unhandledErrors), 1)
        self.assertEqual(self.sender.unhandledErrors[0].value, err)

    def test_unhandledSerializationError(self):
        """
        Errors during serialization ought to be relayed to the sender's
        unhandledError method.
        """
        err = RuntimeError('something undefined went wrong')

        def thunk(result):

            class BrokenBox(amp.Box):

                def _sendTo(self, proto):
                    raise err
            return BrokenBox()
        self.locator.commands['hello'] = thunk
        input = amp.Box(_command='hello', _ask='test-command-id', hello='world')
        self.sender.expectError()
        self.dispatcher.ampBoxReceived(input)
        self.assertEqual(len(self.sender.unhandledErrors), 1)
        self.assertEqual(self.sender.unhandledErrors[0].value, err)

    def test_callRemote(self):
        """
        L{CommandDispatcher.callRemote} should emit a properly formatted '_ask'
        box to its boxSender and record an outstanding L{Deferred}.  When a
        corresponding '_answer' packet is received, the L{Deferred} should be
        fired, and the results translated via the given L{Command}'s response
        de-serialization.
        """
        D = self.dispatcher.callRemote(Hello, hello=b'world')
        self.assertEqual(self.sender.sentBoxes, [amp.AmpBox(_command=b'hello', _ask=b'1', hello=b'world')])
        answers = []
        D.addCallback(answers.append)
        self.assertEqual(answers, [])
        self.dispatcher.ampBoxReceived(amp.AmpBox({b'hello': b'yay', b'print': b'ignored', b'_answer': b'1'}))
        self.assertEqual(answers, [dict(hello=b'yay', Print='ignored')])

    def _localCallbackErrorLoggingTest(self, callResult):
        """
        Verify that C{callResult} completes with a L{None} result and that an
        unhandled error has been logged.
        """
        finalResult = []
        callResult.addBoth(finalResult.append)
        self.assertEqual(1, len(self.sender.unhandledErrors))
        self.assertIsInstance(self.sender.unhandledErrors[0].value, ZeroDivisionError)
        self.assertEqual([None], finalResult)

    def test_callRemoteSuccessLocalCallbackErrorLogging(self):
        """
        If the last callback on the L{Deferred} returned by C{callRemote} (added
        by application code calling C{callRemote}) fails, the failure is passed
        to the sender's C{unhandledError} method.
        """
        self.sender.expectError()
        callResult = self.dispatcher.callRemote(Hello, hello=b'world')
        callResult.addCallback(lambda result: 1 // 0)
        self.dispatcher.ampBoxReceived(amp.AmpBox({b'hello': b'yay', b'print': b'ignored', b'_answer': b'1'}))
        self._localCallbackErrorLoggingTest(callResult)

    def test_callRemoteErrorLocalCallbackErrorLogging(self):
        """
        Like L{test_callRemoteSuccessLocalCallbackErrorLogging}, but for the
        case where the L{Deferred} returned by C{callRemote} fails.
        """
        self.sender.expectError()
        callResult = self.dispatcher.callRemote(Hello, hello=b'world')
        callResult.addErrback(lambda result: 1 // 0)
        self.dispatcher.ampBoxReceived(amp.AmpBox({b'_error': b'1', b'_error_code': b'bugs', b'_error_description': b'stuff'}))
        self._localCallbackErrorLoggingTest(callResult)