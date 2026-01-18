from random import Random
from typing import Awaitable, Dict, List, TypeVar, Union
from hamcrest import (
from hypothesis import given
from hypothesis.strategies import binary, integers, just, lists, randoms, text
from twisted.internet.defer import Deferred, fail
from twisted.internet.interfaces import IProtocol
from twisted.internet.protocol import Protocol
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, connect
from twisted.trial.unittest import SynchronousTestCase
from ..stream import StreamOpen, StreamReceiver, StreamWrite, chunk, stream
from .matchers import HasSum, IsSequenceOf
class InteractTests(SynchronousTestCase):
    """
    Tests for the test helper L{interact}.
    """

    def test_failure(self):
        """
        If the interaction results in a failure then L{interact} raises an
        exception.
        """

        class ArbitraryException(Exception):
            pass
        with self.assertRaises(ArbitraryException):
            interact(Protocol(), Protocol(), fail(ArbitraryException()))

    def test_incomplete(self):
        """
        If the interaction fails to produce a result then L{interact} raises
        an exception.
        """
        with self.assertRaises(Exception):
            interact(Protocol(), Protocol(), Deferred())