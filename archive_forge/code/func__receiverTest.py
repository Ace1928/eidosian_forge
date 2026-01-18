from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def _receiverTest(self, sentences: Iterable[bytes], expectedFired: Iterable[str]=(), extraTest: Callable[[], None] | None=None) -> None:
    """
        A generic test for NMEA receiver behavior.

        @param sentences: The sequence of sentences to simulate receiving.
        @type sentences: iterable of C{str}
        @param expectedFired: The names of the callbacks expected to fire.
        @type expectedFired: iterable of C{str}
        @param extraTest: An optional extra test hook.
        @type extraTest: nullary callable
        """
    for sentence in sentences:
        self.protocol.lineReceived(sentence)
    actuallyFired = self.receiver.called.keys()
    self.assertEqual(set(actuallyFired), set(expectedFired))
    if extraTest is not None:
        extraTest()
    self.receiver.clear()
    self.adapter.clear()