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
def _parserTest(self, sentence: bytes, expected: dict[str, str]) -> None:
    """
        Passes a sentence to the protocol and gets the parsed sentence from
        the receiver. Then verifies that the parsed sentence contains the
        expected data.
        """
    self.protocol.lineReceived(sentence)
    received = self.receiver.receivedSentence
    assert received is not None
    self.assertEqual(expected, received._sentenceData)